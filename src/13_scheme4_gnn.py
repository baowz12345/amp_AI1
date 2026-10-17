# -*- coding: utf-8 -*-
"""
13_scheme4_gnn.py   方案4: GAT/GCN 图学习(手写消息传递层, 零 pyG 依赖)
====================
以氨基酸残基为节点、序列相邻残基为边(路径图)构建样本级图,
节点特征 = one-hot(20) + Kyte-Doolittle 疏水性(1) + 位置(1), 共 22 维。
GCN: 对称归一化邻接聚合; GAT: 加性注意力加权聚合。两者均 2 层 + 均值池化 + MLP。
统一协议: StratifiedKFold 10 折 seed=42, 折内 10% 早停验证, BCE+pos_weight。
输出: results/metrics_S10_GCN.csv / S10_GAT.csv, oof_*.npy
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
HID = 128
OUT = 64

# Kyte-Doolittle 疏水性 (1982, 经典公开值)
KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
      "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
      "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}


def node_features(seqs, max_len=MAX_LEN):
    """one-hot + 疏水 + 位置 = (N, max_len, 22); pad 位置全 0"""
    N = len(seqs)
    X = np.zeros((N, max_len, len(C.AAS) + 2), dtype=np.float32)
    for i, s in enumerate(seqs):
        L = min(len(s), max_len)
        for j in range(L):
            aa = s[j]
            X[i, j, C.AA2IDX[aa]] = 1.0
            X[i, j, 20] = KD[aa]
            X[i, j, 21] = j / max_len
    return X


def build_adj_tpl(L=MAX_LEN):
    """路径图模板: 自环 + 相邻边"""
    A = np.zeros((L, L), dtype=np.float32)
    for i in range(L):
        A[i, i] = 1.0
        if i + 1 < L:
            A[i, i + 1] = 1.0
            A[i + 1, i] = 1.0
    return A


class GCNClassifier(nn.Module):
    def __init__(self, d_in=22, hid=HID, out=OUT):
        super().__init__()
        self.W1 = nn.Linear(d_in, hid, bias=False)
        self.W2 = nn.Linear(hid, out, bias=False)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(out, 64), nn.ReLU(),
                                  nn.Dropout(0.3), nn.Linear(64, 1))
        self.register_buffer("A_tpl", torch.from_numpy(build_adj_tpl()))

    def _norm_adj(self, mask):
        # mask: (B,L) -> 对称归一化邻接 (B,L,L)
        mf = mask.float()
        Am = self.A_tpl * (mf.unsqueeze(1) * mf.unsqueeze(2))
        D = Am.sum(-1).clamp(min=1e-6)
        Ds = D.pow(-0.5)
        return Ds.unsqueeze(-1) * Am * Ds.unsqueeze(-2)

    def forward(self, x):
        mask = (x.abs().sum(-1) > 0)                # (B,L)
        An = self._norm_adj(mask)
        h = F.relu(torch.bmm(An, x) @ self.W1.weight.t())   # (B,L,hid)
        h = F.relu(torch.bmm(An, h) @ self.W2.weight.t())    # (B,L,out)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1).float()
        ctx = (h * mask.unsqueeze(-1).float()).sum(dim=1) / denom
        return self.head(ctx).squeeze(-1)


class GATClassifier(nn.Module):
    """加性注意力图注意力网络(单头 x2 层)"""

    def __init__(self, d_in=22, hid=HID, out=OUT):
        super().__init__()
        self.W1 = nn.Linear(d_in, hid)
        self.W2 = nn.Linear(hid, out)
        self.a1 = nn.Parameter(torch.randn(hid))
        self.a2 = nn.Parameter(torch.randn(hid))
        self.b1 = nn.Parameter(torch.randn(out))
        self.b2 = nn.Parameter(torch.randn(out))
        self.leaky = nn.LeakyReLU(0.2)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(out, 64), nn.ReLU(),
                                  nn.Dropout(0.3), nn.Linear(64, 1))
        self.register_buffer("A_tpl", torch.from_numpy(build_adj_tpl()))

    def _attn_layer(self, x, mask, a1, a2, W, bias):
        # x:(B,L,d) -> h=(B,L,hid); e_ij=LeakyReLU(a1·h_i + a2·h_j)
        h = W(x) + bias                                # (B,L,hid)
        e = self.leaky(h @ a1).unsqueeze(-1) + self.leaky(h @ a2).unsqueeze(-2)  # (B,L,L)
        adj = self.A_tpl * (mask.unsqueeze(1).float() * mask.unsqueeze(2).float())
        e = e.masked_fill(adj == 0, float("-inf"))
        # pad 行(整行 -inf) 改为均匀分布, 避免 softmax NaN(该行输出随后被 mask 池化排除)
        e = torch.where(e.isinf().all(dim=2, keepdim=True), torch.zeros_like(e), e)
        alpha = torch.softmax(e, dim=2)
        return torch.bmm(alpha, h)

    def forward(self, x):
        mask = (x.abs().sum(-1) > 0)
        h = self._attn_layer(x, mask, self.a1, self.a2, self.W1, torch.zeros(self.W1.out_features, device=x.device))
        h = self._attn_layer(h, mask, self.b1, self.b2, self.W2, torch.zeros(self.W2.out_features, device=x.device))
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1).float()
        ctx = (h * mask.unsqueeze(-1).float()).sum(dim=1) / denom
        return self.head(ctx).squeeze(-1)


class FeatDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.FloatTensor(X)
        self.y = None if y is None else torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        if self.y is None:
            return self.X[i]
        return self.X[i], self.y[i]


def train_feat(model, X, y, tr_idx, val_idx, device,
               epochs=EPOCHS, batch=BATCH, lr=LR, patience=PATIENCE, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model.to(device)
    dl_tr = DataLoader(FeatDataset(X[tr_idx], y[tr_idx]), batch_size=batch, shuffle=True, num_workers=0)
    dl_val = DataLoader(FeatDataset(X[val_idx], y[val_idx]), batch_size=batch, shuffle=False, num_workers=0)
    n_pos = int(y[tr_idx].sum())
    n_neg = len(tr_idx) - n_pos
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / max(n_pos, 1)], device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_val, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        losses, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in dl_val:
                xb, yb = xb.to(device), yb.to(device)
                losses += loss_fn(model(xb), yb).item() * len(xb)
                n += len(xb)
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
def predict_feat(model, X, te_idx, device, batch=BATCH):
    model.eval()
    dl = DataLoader(FeatDataset(X[te_idx]), batch_size=batch, shuffle=False)
    probs = []
    for xb in dl:
        xb = xb.to(device)
        probs.append(torch.sigmoid(model(xb)).cpu().numpy())
    oof = np.concatenate(probs)
    out = np.zeros(len(X))
    out[te_idx] = oof
    return out


def run_one(model_cls, name, X, y, device):
    skf = C.stratified_kfold()
    fold_metrics, oof = [], np.zeros(len(y))
    for fold, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        from sklearn.model_selection import train_test_split
        tr_sub, val_sub = train_test_split(np.arange(len(tr_idx)), test_size=0.1,
                                           stratify=y[tr_idx], random_state=CV_FOLD_SEED + fold)
        tr2, val = tr_idx[tr_sub], tr_idx[val_sub]
        model = model_cls()
        model = train_feat(model, X, y, tr2, val, device, seed=CV_FOLD_SEED + fold)
        prob = predict_feat(model, X, te_idx, device)
        oof[te_idx] = prob[te_idx]
        fold_metrics.append(C.compute_metrics(y[te_idx], prob[te_idx]))
        print(f"  折 {fold}/10 完成 (AUC={fold_metrics[-1]['AUC']:.4f})")
    df_folds = pd.DataFrame(fold_metrics)
    df_folds.insert(0, "fold", np.arange(1, C.N_FOLDS + 1))
    pooled = C.compute_metrics(y, oof)
    df_out = pd.concat([df_folds, pd.DataFrame([{**pooled, "fold": "pooled_OOF"}])], ignore_index=True)
    df_out.insert(0, "model", name)
    df_out.to_csv(os.path.join(C.RESULT_DIR, f"metrics_{name}.csv"), index=False, encoding="utf-8-sig")
    np.save(os.path.join(C.RESULT_DIR, f"oof_{name}.npy"), oof)
    print(f"[{name}] AUC={pooled['AUC']:.4f} PR_AUC={pooled['PR_AUC']:.4f} MCC={pooled['MCC']:.4f} "
          f"Acc={pooled['Acc']:.4f} F1={pooled['F1']:.4f}")
    return df_out, oof


def main():
    C.setup_chinese_font()
    device = C.get_device()
    print(f"设备: {device}")
    df = C.load_dataset()
    seqs, y = df["seq"].tolist(), df["label"].values
    X = node_features(seqs)
    print(f"节点特征: (N, {MAX_LEN}, {X.shape[2]})  边=序列相邻残基")

    run_one(GCNClassifier, "S10_GCN", X, y, device)
    run_one(GATClassifier, "S10_GAT", X, y, device)


if __name__ == "__main__":
    main()
