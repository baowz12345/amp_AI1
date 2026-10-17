# -*- coding: utf-8 -*-
"""
dl_common.py
============
深度学习公共模块: 数据集封装、通用训练器(早停)、CNN / BiLSTM+Attention / Transformer
三种序列分类网络。输入均为索引编码 (N, MAX_LEN), pad 位置索引 = PAD_IDX。
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import common as C

MAX_LEN = 80
EPOCHS = 40
BATCH = 64
LR = 1e-3
PATIENCE = 8
EMB_DIM = 64


class SeqDataset(Dataset):
    def __init__(self, X, y=None):
        if np.issubdtype(X.dtype, np.floating):
            self.X = torch.FloatTensor(X)
        else:
            self.X = torch.LongTensor(X)
        self.y = None if y is None else torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        if self.y is None:
            return self.X[i]
        return self.X[i], self.y[i]


def pad_mask(x):
    """x: (B, L) 索引 -> mask (B, L) True=有效位置"""
    return x != C.PAD_IDX


# ---------------------------------------------------------------------------
# 网络
# ---------------------------------------------------------------------------
class CNNClassifier(nn.Module):
    """一维卷积网络: 输入 one-hot (B, L, 20)"""

    def __init__(self, n_aa=20, n_ch1=64, n_ch2=128, n_ch3=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_aa, n_ch1, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(n_ch1, n_ch2, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(n_ch2, n_ch3, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(n_ch3, 64), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(64, 1),
        )

    def forward(self, x_oh):
        x = x_oh.permute(0, 2, 1)          # (B,20,L)
        x = self.conv(x)                    # (B,128,L//8)
        x = x.mean(dim=2)                   # GAP
        return self.head(x).squeeze(-1)


class AttentionPool(nn.Module):
    """基于有效位置的加性注意力池化"""

    def __init__(self, d_in):
        super().__init__()
        self.W = nn.Linear(d_in, d_in)
        self.u = nn.Parameter(torch.randn(d_in))

    def forward(self, h, mask):
        scores = torch.tanh(self.W(h)) @ self.u   # (B,L)
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(scores, dim=1).unsqueeze(1)  # (B,1,L)
        return torch.bmm(attn, h).squeeze(1)               # (B,d)


class BiLSTMAttnClassifier(nn.Module):
    """BiLSTM + 注意力池化"""

    def __init__(self, vocab, emb=EMB_DIM, hid=64, layers=2, n_out=1):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=C.PAD_IDX)
        self.lstm = nn.LSTM(emb, hid, num_layers=layers, batch_first=True,
                            bidirectional=True, dropout=0.2)
        self.attn = AttentionPool(hid * 2)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(hid * 2, 32),
                                  nn.ReLU(), nn.Linear(32, n_out))

    def forward(self, x):
        mask = pad_mask(x)
        h = self.emb(x)
        h, _ = self.lstm(h)
        ctx = self.attn(h, mask)
        return self.head(ctx).squeeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerClassifier(nn.Module):
    """Transformer 编码器 + 均值池化"""

    def __init__(self, vocab, emb=EMB_DIM, heads=4, layers=2, ff=128, n_out=1):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=C.PAD_IDX)
        self.pos = PositionalEncoding(emb)
        enc_layer = nn.TransformerEncoderLayer(d_model=emb, nhead=heads,
                                               dim_feedforward=ff,
                                               dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(emb, 32),
                                  nn.ReLU(), nn.Linear(32, n_out))

    def forward(self, x):
        mask = pad_mask(x)
        h = self.emb(x) * (EMB_DIM ** 0.5)
        h = self.pos(h)
        h = self.encoder(h, src_key_padding_mask=~mask)
        # 有效位置均值池化
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1).float()
        ctx = (h * mask.unsqueeze(-1).float()).sum(dim=1) / denom
        return self.head(ctx).squeeze(-1)


# ---------------------------------------------------------------------------
# 训练器
# ---------------------------------------------------------------------------
def train_model(model, X_all, y_all, tr_idx, val_idx, device,
                epochs=EPOCHS, batch=BATCH, lr=LR, patience=PATIENCE, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model.to(device)
    ds_tr = SeqDataset(X_all[tr_idx], y_all[tr_idx])
    ds_val = SeqDataset(X_all[val_idx], y_all[val_idx])
    dl_tr = DataLoader(ds_tr, batch_size=batch, shuffle=True, num_workers=0)
    dl_val = DataLoader(ds_val, batch_size=batch, shuffle=False, num_workers=0)

    n_pos = int(y_all[tr_idx].sum())
    n_neg = len(tr_idx) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        # 验证
        model.eval()
        losses, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in dl_val:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                losses += loss_fn(out, yb).item() * len(xb)
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
def predict_prob(model, X_all, te_idx, device, batch=BATCH):
    model.eval()
    ds = SeqDataset(X_all[te_idx])
    dl = DataLoader(ds, batch_size=batch, shuffle=False)
    probs = []
    for xb in dl:
        xb = xb.to(device)
        out = model(xb)
        probs.append(torch.sigmoid(out).cpu().numpy())
    oof = np.concatenate(probs)
    out_full = np.zeros(len(X_all))
    out_full[te_idx] = oof
    return out_full


def build_model(model_name, vocab):
    if model_name == "CNN":
        return CNNClassifier()
    if model_name == "BiLSTM":
        return BiLSTMAttnClassifier(vocab)
    if model_name == "Transformer":
        return TransformerClassifier(vocab)
    raise ValueError(model_name)


# ---------------------------------------------------------------------------
# 10折 CV 运行器 (与 sklearn 方案共用同一折划分)
# ---------------------------------------------------------------------------
def run_deep_cv(model_name, X, y, scheme_prefix="S4", device=None):
    from sklearn.model_selection import train_test_split
    if device is None:
        device = C.get_device()
    print(f"  [{model_name}] 设备: {device}")
    skf = C.stratified_kfold()
    vocab = len(C.AAS) + 1
    fold_metrics, oof = [], np.zeros(len(y))
    for fold, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        # 训练折内再划 10% 作早停验证
        tr_sub, val_sub = train_test_split(np.arange(len(tr_idx)), test_size=0.1,
                                           stratify=y[tr_idx], random_state=CV_FOLD_SEED + fold)
        tr_idx2, val_idx = tr_idx[tr_sub], tr_idx[val_sub]
        model = build_model(model_name, vocab)
        model = train_model(model, X, y, tr_idx2, val_idx, device, seed=CV_FOLD_SEED + fold)
        prob = predict_prob(model, X, te_idx, device)
        oof[te_idx] = prob[te_idx]
        fold_metrics.append(C.compute_metrics(y[te_idx], prob[te_idx]))
        print(f"    折 {fold}/10 完成 (AUC={fold_metrics[-1]['AUC']:.4f})")
    full = f"{scheme_prefix}_{model_name}"
    df_folds = pd.DataFrame(fold_metrics)
    df_folds.insert(0, "fold", np.arange(1, C.N_FOLDS + 1))
    pooled = C.compute_metrics(y, oof)
    df_out = pd.concat([df_folds, pd.DataFrame([{**pooled, "fold": "pooled_OOF"}])],
                       ignore_index=True)
    df_out.insert(0, "model", full)
    df_out.to_csv(os.path.join(C.RESULT_DIR, f"metrics_{full}.csv"), index=False, encoding="utf-8-sig")
    np.save(os.path.join(C.RESULT_DIR, f"oof_{full}.npy"), oof)
    line = {"方案": scheme_prefix, "模型": model_name}
    line.update(C.format_metric(df_folds))
    print(f"  [{model_name}] AUC={line['AUC']}  F1={line['F1']}  MCC={line['MCC']}")
    return df_out, oof


CV_FOLD_SEED = 1000
