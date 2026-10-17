# -*- coding: utf-8 -*-
"""
16_scheme9_multimodal.py   方案9: 序列 + 二级结构 多模态融合
===========================
引入结构生物学先验: 用 Chou-Fasman 经典倾向参数(公开文献值)对每条序列做启发式
SS3 预测(H=螺旋/E=折叠/C=卷曲), 得到二级结构 one-hot 序列, 与氨基酸序列嵌入
双分支融合(序列 BiGRU + 结构 Conv1D -> concat -> MLP)后二分类。
统一协议: StratifiedKFold 10 折 seed=42, 训练折内 10% 早停验证,
BCEWithLogitsLoss + pos_weight, 输出 OOF 概率与全套指标。
输出: results/metrics_S14_MultiModal.csv, results/oof_S14_MultiModal.npy
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

MAX_LEN = 80
EPOCHS = 40
BATCH = 64
LR = 1e-3
PATIENCE = 8
CV_FOLD_SEED = 1000

# Chou-Fasman 倾向参数 (P_alpha, P_beta, P_turn) 经典公开值
CF_TABLE = {
    "A": (142, 83, 66), "R": (98, 93, 95), "N": (67, 89, 156), "D": (101, 54, 146),
    "C": (70, 119, 119), "Q": (111, 110, 98), "E": (151, 37, 74), "G": (57, 75, 156),
    "H": (100, 87, 95), "I": (108, 160, 47), "L": (121, 130, 59), "K": (114, 74, 101),
    "M": (145, 105, 57), "F": (113, 138, 60), "P": (57, 55, 152), "S": (77, 75, 143),
    "T": (82, 119, 96), "W": (108, 137, 96), "Y": (69, 147, 114), "V": (106, 170, 50),
}
SS3_NAMES = ["H", "E", "C"]


def predict_ss3(seq, win=5):
    """窗口平滑的启发式 SS3 预测: 每残基取窗口内平均倾向最高的类别"""
    n = len(seq)
    out = []
    for i in range(n):
        lo, hi = max(0, i - win // 2), min(n, i + win // 2 + 1)
        sums = [0.0, 0.0, 0.0]
        for j in range(lo, hi):
            p = CF_TABLE[seq[j]]
            for k in range(3):
                sums[k] += p[k]
        k = int(np.argmax(sums))
        out.append(SS3_NAMES[k])
    return "".join(out)


def ss3_onehot(seqs, max_len=MAX_LEN):
    """SS3 序列 -> (N, max_len, 3) one-hot"""
    ss2idx = {s: i for i, s in enumerate(SS3_NAMES)}
    X = np.zeros((len(seqs), max_len, 3), dtype=np.float32)
    for i, s in enumerate(seqs):
        ss = predict_ss3(s)[:max_len]
        for j, ch in enumerate(ss):
            X[i, j, ss2idx[ch]] = 1.0
    return X


class PairDataset(Dataset):
    def __init__(self, x_idx, x_ss, y=None):
        self.x1 = torch.LongTensor(x_idx)
        self.x2 = torch.FloatTensor(x_ss)
        self.y = None if y is None else torch.FloatTensor(y)

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, i):
        if self.y is None:
            return self.x1[i], self.x2[i]
        return self.x1[i], self.x2[i], self.y[i]


class MultiModalClassifier(nn.Module):
    """序列(BiGRU) + 二级结构(Conv1D) 双分支融合"""

    def __init__(self, vocab, emb=64, hid=64, ss_conv=32):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=C.PAD_IDX)
        self.gru = nn.GRU(emb, hid, batch_first=True, bidirectional=True)
        # 结构分支
        self.ss_conv = nn.Sequential(
            nn.Conv1d(3, ss_conv, 3, padding=1), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(hid * 2 + ss_conv, 64), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(64, 1),
        )

    def forward(self, x_idx, x_ss):
        mask = x_idx != C.PAD_IDX                     # (B,L)
        h = self.emb(x_idx)                            # (B,L,emb)
        h, _ = self.gru(h)                             # (B,L,2*hid)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1).float()
        h_seq = (h * mask.unsqueeze(-1).float()).sum(dim=1) / denom   # (B,2*hid)
        s = x_ss.permute(0, 2, 1)                      # (B,3,L)
        s = self.ss_conv(s)                            # (B,ss_conv,L)
        h_ss = (s * mask.unsqueeze(1).float()).sum(dim=2) / denom     # (B,ss_conv)
        return self.head(torch.cat([h_seq, h_ss], dim=1)).squeeze(-1)


def train_pair(model, X_idx, X_ss, y, tr_idx, val_idx, device,
               epochs=EPOCHS, batch=BATCH, lr=LR, patience=PATIENCE, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model.to(device)
    ds_tr = PairDataset(X_idx[tr_idx], X_ss[tr_idx], y[tr_idx])
    ds_val = PairDataset(X_idx[val_idx], X_ss[val_idx], y[val_idx])
    dl_tr = DataLoader(ds_tr, batch_size=batch, shuffle=True, num_workers=0)
    dl_val = DataLoader(ds_val, batch_size=batch, shuffle=False, num_workers=0)

    n_pos = int(y[tr_idx].sum())
    n_neg = len(tr_idx) - n_pos
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / max(n_pos, 1)], device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        for xb, sb, yb in dl_tr:
            xb, sb, yb = xb.to(device), sb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb, sb)
            loss = loss_fn(out, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        losses, n = 0.0, 0
        with torch.no_grad():
            for xb, sb, yb in dl_val:
                xb, sb, yb = xb.to(device), sb.to(device), yb.to(device)
                losses += loss_fn(model(xb, sb), yb).item() * len(xb)
                n += len(xb)
        vloss = losses / n
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_pair(model, X_idx, X_ss, te_idx, device, batch=BATCH):
    model.eval()
    ds = PairDataset(X_idx[te_idx], X_ss[te_idx])
    dl = DataLoader(ds, batch_size=batch, shuffle=False)
    probs = []
    for xb, sb in dl:
        xb, sb = xb.to(device), sb.to(device)
        probs.append(torch.sigmoid(model(xb, sb)).cpu().numpy())
    oof = np.concatenate(probs)
    out_full = np.zeros(len(X_idx))
    out_full[te_idx] = oof
    return out_full


def main():
    C.setup_chinese_font()
    device = C.get_device()
    print(f"设备: {device}")
    df = C.load_dataset()
    seqs, y = df["seq"].tolist(), df["label"].values
    X_idx = C.seq_to_indices(seqs, MAX_LEN)
    X_ss = ss3_onehot(seqs, MAX_LEN)
    print(f"SS3 分布: H={(X_ss[:, :, 0].sum()):.0f} E={(X_ss[:, :, 1].sum()):.0f} C={(X_ss[:, :, 2].sum()):.0f}")

    skf = C.stratified_kfold()
    vocab = len(C.AAS) + 1
    fold_metrics, oof = [], np.zeros(len(y))
    for fold, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        from sklearn.model_selection import train_test_split
        tr_sub, val_sub = train_test_split(np.arange(len(tr_idx)), test_size=0.1,
                                           stratify=y[tr_idx], random_state=CV_FOLD_SEED + fold)
        tr2, val = tr_idx[tr_sub], tr_idx[val_sub]
        model = MultiModalClassifier(vocab)
        model = train_pair(model, X_idx, X_ss, y, tr2, val, device, seed=CV_FOLD_SEED + fold)
        prob = predict_pair(model, X_idx, X_ss, te_idx, device)
        oof[te_idx] = prob[te_idx]
        fold_metrics.append(C.compute_metrics(y[te_idx], prob[te_idx]))
        print(f"  折 {fold}/10 完成 (AUC={fold_metrics[-1]['AUC']:.4f})")

    full = "S14_MultiModal"
    df_folds = pd.DataFrame(fold_metrics)
    df_folds.insert(0, "fold", np.arange(1, C.N_FOLDS + 1))
    pooled = C.compute_metrics(y, oof)
    df_out = pd.concat([df_folds, pd.DataFrame([{**pooled, "fold": "pooled_OOF"}])], ignore_index=True)
    df_out.insert(0, "model", full)
    df_out.to_csv(os.path.join(C.RESULT_DIR, f"metrics_{full}.csv"), index=False, encoding="utf-8-sig")
    np.save(os.path.join(C.RESULT_DIR, f"oof_{full}.npy"), oof)
    print(f"[{full}] AUC={pooled['AUC']:.4f} PR_AUC={pooled['PR_AUC']:.4f} MCC={pooled['MCC']:.4f} "
          f"Acc={pooled['Acc']:.4f} F1={pooled['F1']:.4f}")


if __name__ == "__main__":
    main()
