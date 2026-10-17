# -*- coding: utf-8 -*-
"""
15_scheme8_moe.py   方案8: MoE 混合专家(门控按肽长度/理化分治)
====================
共享序列编码(Embedding+GRU 均值池化)得到 h_seq, 拼接全局理化特征(len/电荷/疏水/pI),
门控网络 softmax 将其路由到 K=3 个专家 MLP, 输出为专家输出的门控加权和。
统一协议: StratifiedKFold 10 折 seed=42, 折内 10% 早停验证, BCE+pos_weight。
输出: results/metrics_S13_MoE.csv, results/oof_S13_MoE.npy
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

MAX_LEN = 80
EPOCHS = 40
BATCH = 64
LR = 1e-3
PATIENCE = 8
CV_FOLD_SEED = 1000
N_EXPERTS = 3
GLOB_NAMES = ["len", "charge_pH7", "hydrophobicity", "pI"]
GLOB_SCALE = np.array([100.0, 10.0, 5.0, 14.0], dtype=np.float32)  # 粗略归一化


def global_features(seqs):
    """从 physico_features 中取 4 维全局理化特征并归一化"""
    X15, _names = C.physico_features(seqs)
    idxs = [0, 2, 4, 3]   # len, charge_pH7, hydrophobicity, pI
    Xg = X15[:, idxs].astype(np.float32) / GLOB_SCALE
    return Xg


class MoEClassifier(nn.Module):
    """门控路由 + K 专家"""

    def __init__(self, vocab, emb=64, hid=64, k=N_EXPERTS, n_glob=4):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=C.PAD_IDX)
        self.gru = nn.GRU(emb, hid, batch_first=True, bidirectional=True)
        gate_in = hid * 2 + n_glob
        self.gate = nn.Sequential(nn.Linear(gate_in, 32), nn.ReLU(),
                                  nn.Linear(32, k), nn.Softmax(dim=1))
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Dropout(0.3), nn.Linear(hid * 2, 32), nn.ReLU(),
                          nn.Dropout(0.2), nn.Linear(32, 1)) for _ in range(k)])

    def forward(self, x_idx, xg):
        mask = x_idx != C.PAD_IDX
        h = self.emb(x_idx)
        h, _ = self.gru(h)                                  # (B,L,2*hid)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1).float()
        h_seq = (h * mask.unsqueeze(-1).float()).sum(dim=1) / denom   # (B,2*hid)
        g_in = torch.cat([h_seq, xg], dim=1)
        g = self.gate(g_in)                                 # (B,K)
        outs = torch.stack([e(h_seq) for e in self.experts], dim=1)   # (B,K,1)
        out = (g.unsqueeze(-1) * outs).sum(dim=1).squeeze(-1)         # (B,)
        return out, g


class MoEDataset(Dataset):
    def __init__(self, x_idx, xg, y=None):
        self.x1 = torch.LongTensor(x_idx)
        self.x2 = torch.FloatTensor(xg)
        self.y = None if y is None else torch.FloatTensor(y)

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, i):
        if self.y is None:
            return self.x1[i], self.x2[i]
        return self.x1[i], self.x2[i], self.y[i]


def train_moe(model, X_idx, Xg, y, tr_idx, val_idx, device,
              epochs=EPOCHS, batch=BATCH, lr=LR, patience=PATIENCE, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model.to(device)
    dl_tr = DataLoader(MoEDataset(X_idx[tr_idx], Xg[tr_idx], y[tr_idx]),
                       batch_size=batch, shuffle=True, num_workers=0)
    dl_val = DataLoader(MoEDataset(X_idx[val_idx], Xg[val_idx], y[val_idx]),
                        batch_size=batch, shuffle=False, num_workers=0)
    n_pos = int(y[tr_idx].sum())
    n_neg = len(tr_idx) - n_pos
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / max(n_pos, 1)], device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_val, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        for ib, gb, yb in dl_tr:
            ib, gb, yb = ib.to(device), gb.to(device), yb.to(device)
            opt.zero_grad()
            out, _ = model(ib, gb)
            loss = loss_fn(out, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        losses, n = 0.0, 0
        with torch.no_grad():
            for ib, gb, yb in dl_val:
                ib, gb, yb = ib.to(device), gb.to(device), yb.to(device)
                out, _ = model(ib, gb)
                losses += loss_fn(out, yb).item() * len(ib)
                n += len(ib)
        vloss = losses / n
        if vloss < best_val:
            best_val, best_state = vloss, {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_moe(model, X_idx, Xg, te_idx, device, batch=BATCH):
    model.eval()
    dl = DataLoader(MoEDataset(X_idx[te_idx], Xg[te_idx]), batch_size=batch, shuffle=False)
    probs, gates = [], []
    for ib, gb in dl:
        ib, gb = ib.to(device), gb.to(device)
        out, g = model(ib, gb)
        probs.append(torch.sigmoid(out).cpu().numpy())
        gates.append(g.cpu().numpy())
    oof = np.concatenate(probs)
    out_full = np.zeros(len(X_idx))
    out_full[te_idx] = oof
    return out_full, np.concatenate(gates)


def main():
    C.setup_chinese_font()
    device = C.get_device()
    print(f"设备: {device}")
    df = C.load_dataset()
    seqs, y = df["seq"].tolist(), df["label"].values
    X_idx = C.seq_to_indices(seqs, MAX_LEN)
    Xg = global_features(seqs)
    print(f"全局特征: {GLOB_NAMES}  专家数 K={N_EXPERTS}")

    skf = C.stratified_kfold()
    vocab = len(C.AAS) + 1
    fold_metrics, oof, gate_all = [], np.zeros(len(y)), []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        from sklearn.model_selection import train_test_split
        tr_sub, val_sub = train_test_split(np.arange(len(tr_idx)), test_size=0.1,
                                           stratify=y[tr_idx], random_state=CV_FOLD_SEED + fold)
        tr2, val = tr_idx[tr_sub], tr_idx[val_sub]
        model = MoEClassifier(vocab)
        model = train_moe(model, X_idx, Xg, y, tr2, val, device, seed=CV_FOLD_SEED + fold)
        prob, g = predict_moe(model, X_idx, Xg, te_idx, device)
        oof[te_idx] = prob[te_idx]
        gate_all.append(g)
        fold_metrics.append(C.compute_metrics(y[te_idx], prob[te_idx]))
        print(f"  折 {fold}/10 完成 (AUC={fold_metrics[-1]['AUC']:.4f})")

    gates = np.concatenate(gate_all)

    # 先保存主结果(避免后续统计问题丢失 10 折产出)
    full = "S13_MoE"
    df_folds = pd.DataFrame(fold_metrics)
    df_folds.insert(0, "fold", np.arange(1, C.N_FOLDS + 1))
    pooled = C.compute_metrics(y, oof)
    df_out = pd.concat([df_folds, pd.DataFrame([{**pooled, "fold": "pooled_OOF"}])], ignore_index=True)
    df_out.insert(0, "model", full)
    df_out.to_csv(os.path.join(C.RESULT_DIR, f"metrics_{full}.csv"), index=False, encoding="utf-8-sig")
    np.save(os.path.join(C.RESULT_DIR, f"oof_{full}.npy"), oof)

    # 门控分布统计(按预测长度分桶展示路由偏好)
    try:
        lens = df["len"].values
        bins = pd.cut(lens, bins=3, labels=["短肽<20", "中肽20-40", "长肽>40"])
        gdf = pd.DataFrame(gates, columns=[f"专家{k+1}" for k in range(N_EXPERTS)])
        gdf["长度桶"] = np.asarray(bins)
        gdf["label"] = y
        print("\n门控平均权重(按长度桶):")
        print(gdf.groupby("长度桶")[[f"专家{k+1}" for k in range(N_EXPERTS)]].mean().round(3).to_string())
        gdf.to_csv(os.path.join(C.RESULT_DIR, "moe_gate_stats.csv"), index=False, encoding="utf-8-sig")
    except Exception as e:
        print(f"门控统计跳过: {e}")
    print(f"[{full}] AUC={pooled['AUC']:.4f} PR_AUC={pooled['PR_AUC']:.4f} MCC={pooled['MCC']:.4f} "
          f"Acc={pooled['Acc']:.4f} F1={pooled['F1']:.4f}")


if __name__ == "__main__":
    main()
