# -*- coding: utf-8 -*-
"""
11_scheme2_cnn_bilstm.py   方案2: CNN-BiLSTM 端到端融合
==========================
端到端自动学习局部 motif(CNN) 并建模双向上下文(BiLSTM), 注意力池化后分类。
输入: one-hot 序列 + 索引(用于 padding mask)。无需任何手工特征。
统一协议: StratifiedKFold 10 折 seed=42, 折内 10% 早停验证, BCE+pos_weight。
输出: results/metrics_S8_CNNBiLSTM.csv, results/oof_S8_CNNBiLSTM.npy
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


def _pool_mask(m):
    """MaxPool(2) 对应的 padding mask 下采样: 池化窗内任一位置有效则有效"""
    L = m.size(1)
    if L % 2 == 0:
        return m[:, ::2] | m[:, 1::2]
    return m[:, ::2] | m[:, 1::2] | m[:, -1:]


class CNNBiLSTMClassifier(nn.Module):
    """Conv1D 提取局部 motif -> BiLSTM 双向上下文 -> 注意力池化 -> MLP"""

    def __init__(self, n_aa=20, c1=64, c2=128, hid=64, layers=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_aa, c1, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(c1, c2, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(c2, hid, num_layers=layers, batch_first=True,
                            bidirectional=True, dropout=0.2)
        self.attn = C_dl_attn(hid * 2)
        self.head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(hid * 2, 64), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(64, 1),
        )

    def forward(self, x_oh, x_idx):
        # padding mask 随 MaxPool 下采样
        mask0 = (x_idx != C.PAD_IDX)          # (B,L)
        m1 = _pool_mask(mask0)
        m2 = _pool_mask(m1)
        x = x_oh.permute(0, 2, 1)             # (B,20,L)
        x = self.conv(x)                      # (B,c2,L//4)
        x = x.permute(0, 2, 1)                # (B,L//4,c2)
        h, _ = self.lstm(x)                   # (B,L//4,2*hid)
        ctx = self.attn(h, m2)
        return self.head(ctx).squeeze(-1)


class C_dl_attn(nn.Module):
    """加性注意力池化(复用 dl_common 设计)"""

    def __init__(self, d_in):
        super().__init__()
        self.W = nn.Linear(d_in, d_in)
        self.u = nn.Parameter(torch.randn(d_in))

    def forward(self, h, mask):
        scores = torch.tanh(self.W(h)) @ self.u
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(scores, dim=1).unsqueeze(1)
        return torch.bmm(attn, h).squeeze(1)


class DualDataset(Dataset):
    def __init__(self, x_oh, x_idx, y=None):
        self.x_oh = torch.FloatTensor(x_oh)
        self.x_idx = torch.LongTensor(x_idx)
        self.y = None if y is None else torch.FloatTensor(y)

    def __len__(self):
        return len(self.x_oh)

    def __getitem__(self, i):
        if self.y is None:
            return self.x_oh[i], self.x_idx[i]
        return self.x_oh[i], self.x_idx[i], self.y[i]


def train_dual(model, X_oh, X_idx, y, tr_idx, val_idx, device,
               epochs=EPOCHS, batch=BATCH, lr=LR, patience=PATIENCE, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model.to(device)
    dl_tr = DataLoader(DualDataset(X_oh[tr_idx], X_idx[tr_idx], y[tr_idx]),
                       batch_size=batch, shuffle=True, num_workers=0)
    dl_val = DataLoader(DualDataset(X_oh[val_idx], X_idx[val_idx], y[val_idx]),
                        batch_size=batch, shuffle=False, num_workers=0)
    n_pos = int(y[tr_idx].sum())
    n_neg = len(tr_idx) - n_pos
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / max(n_pos, 1)], device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_val, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        for ob, ib, yb in dl_tr:
            ob, ib, yb = ob.to(device), ib.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(ob, ib), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        losses, n = 0.0, 0
        with torch.no_grad():
            for ob, ib, yb in dl_val:
                ob, ib, yb = ob.to(device), ib.to(device), yb.to(device)
                losses += loss_fn(model(ob, ib), yb).item() * len(ob)
                n += len(ob)
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
def predict_dual(model, X_oh, X_idx, te_idx, device, batch=BATCH):
    model.eval()
    dl = DataLoader(DualDataset(X_oh[te_idx], X_idx[te_idx]), batch_size=batch, shuffle=False)
    probs = []
    for ob, ib in dl:
        ob, ib = ob.to(device), ib.to(device)
        probs.append(torch.sigmoid(model(ob, ib)).cpu().numpy())
    oof = np.concatenate(probs)
    out = np.zeros(len(X_oh))
    out[te_idx] = oof
    return out


def main():
    C.setup_chinese_font()
    device = C.get_device()
    print(f"设备: {device}")
    df = C.load_dataset()
    seqs, y = df["seq"].tolist(), df["label"].values
    X_oh = C.onehot_encoding(seqs, MAX_LEN)
    X_idx = C.seq_to_indices(seqs, MAX_LEN)

    skf = C.stratified_kfold()
    vocab = len(C.AAS) + 1
    fold_metrics, oof = [], np.zeros(len(y))
    for fold, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        from sklearn.model_selection import train_test_split
        tr_sub, val_sub = train_test_split(np.arange(len(tr_idx)), test_size=0.1,
                                           stratify=y[tr_idx], random_state=CV_FOLD_SEED + fold)
        tr2, val = tr_idx[tr_sub], tr_idx[val_sub]
        model = CNNBiLSTMClassifier()
        model = train_dual(model, X_oh, X_idx, y, tr2, val, device, seed=CV_FOLD_SEED + fold)
        prob = predict_dual(model, X_oh, X_idx, te_idx, device)
        oof[te_idx] = prob[te_idx]
        fold_metrics.append(C.compute_metrics(y[te_idx], prob[te_idx]))
        print(f"  折 {fold}/10 完成 (AUC={fold_metrics[-1]['AUC']:.4f})")

    full = "S8_CNNBiLSTM"
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
