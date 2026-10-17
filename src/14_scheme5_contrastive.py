# -*- coding: utf-8 -*-
"""
14_scheme5_contrastive.py   方案5: 对比自监督预训练(小样本友好)
============================
利用两个 FASTA 全量无标注肽序列(合并清洗去重)预训练序列编码器:
  增强(随机截断/替换) -> 双视图 -> 共享 BiGRU 编码器 -> 投影头 -> InfoNCE 对比损失。
预训练后冻结编码器, 在 APD 标注数据上做线性探测(LR/SVM), 10 折 CV 评估。
统一协议: StratifiedKFold 10 折 seed=42, 输出 OOF 概率与全套指标。
输出: results/metrics_S11_Contrastive_*.csv, oof_*.npy, results/contrastive_rep.npy
"""
import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

MAX_LEN = 80
PRETRAIN_EPOCHS = 15
BATCH = 128
LR = 1e-3
TAU = 0.1
EMB = 64
HID = 64
CV_FOLD_SEED = 1000
AAS = C.AAS


def parse_fasta(path):
    seqs = []
    cur = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur))
                    cur = []
            else:
                cur.append(line)
    if cur:
        seqs.append("".join(cur))
    return seqs


def clean_seq(s):
    if not s or len(s) < 4 or len(s) > 100:
        return None
    if any(c not in AAS for c in s):
        return None
    return s


def build_unlabeled_pool():
    """合并 animal+natural FASTA, 清洗(20aa/4-100aa), 去重 -> 无标注池"""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths = [os.path.join(root, "animalAMPs_APD2024a.fasta"),
             os.path.join(root, "naturalAMPs_APD2024a.fasta")]
    pool = []
    for p in paths:
        if os.path.exists(p):
            for s in parse_fasta(p):
                c = clean_seq(s)
                if c:
                    pool.append(c)
    pool = sorted(set(pool))
    print(f"无标注肽序列池: {len(pool)} 条")
    return pool


def augment(seq, rng):
    """序列增强: 随机截断(60-100%) + 随机替换 15% 位置"""
    s = seq
    L = len(s)
    keep = rng.randint(max(4, int(0.6 * L)), L)
    start = rng.randint(0, L - keep)
    s = s[start:start + keep]
    if len(s) < 4:
        s = seq
    s = list(s)
    n_rep = max(1, int(0.15 * len(s)))
    for _ in range(n_rep):
        i = rng.randrange(len(s))
        s[i] = AAS[rng.randrange(len(AAS))]
    return "".join(s)


class Encoder(nn.Module):
    """BiGRU 编码器 + 投影头"""

    def __init__(self, vocab, emb=EMB, hid=HID):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=C.PAD_IDX)
        self.gru = nn.GRU(emb, hid, batch_first=True, bidirectional=True)
        self.proj = nn.Sequential(nn.Linear(hid * 2, hid * 2), nn.ReLU(),
                                  nn.Linear(hid * 2, hid * 2))

    def encode(self, x_idx):
        mask = x_idx != C.PAD_IDX
        h = self.emb(x_idx)
        h, _ = self.gru(h)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1).float()
        return (h * mask.unsqueeze(-1).float()).sum(dim=1) / denom   # (B,2*hid)

    def forward(self, x_idx):
        return self.proj(self.encode(x_idx))


class SeqIdxDataset(Dataset):
    def __init__(self, X):
        self.X = torch.LongTensor(X)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i]


def pretrain(encoder, pool_seqs, device, epochs=PRETRAIN_EPOCHS):
    """SimCLR 风格 InfoNCE 预训练"""
    rng = random.Random(0)
    aug1 = [augment(s, rng) for s in pool_seqs]
    aug2 = [augment(s, rng) for s in pool_seqs]
    X1 = C.seq_to_indices(aug1, MAX_LEN)
    X2 = C.seq_to_indices(aug2, MAX_LEN)
    dl1 = DataLoader(SeqIdxDataset(X1), batch_size=BATCH, shuffle=False, num_workers=0)
    dl2 = DataLoader(SeqIdxDataset(X2), batch_size=BATCH, shuffle=False, num_workers=0)
    opt = torch.optim.AdamW(encoder.parameters(), lr=LR, weight_decay=1e-4)
    n_batch = len(dl1)
    for ep in range(epochs):
        encoder.train()
        total = 0.0
        for x1, x2 in zip(dl1, dl2):
            x1, x2 = x1.to(device), x2.to(device)
            z1, z2 = F.normalize(encoder(x1), dim=1), F.normalize(encoder(x2), dim=1)
            sim = z1 @ z2.t() / TAU                     # (B,B)
            labels = torch.arange(len(x1), device=device)
            loss = F.cross_entropy(sim, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"  预训练 epoch {ep + 1}/{epochs}  InfoNCE loss={total / n_batch:.4f}")
    return encoder


@torch.no_grad()
def extract_rep(encoder, seqs, device, batch=BATCH):
    X = C.seq_to_indices(seqs, MAX_LEN)
    dl = DataLoader(SeqIdxDataset(X), batch_size=batch, shuffle=False)
    encoder.eval()
    reps = []
    for xb in dl:
        xb = xb.to(device)
        reps.append(encoder.encode(xb).cpu().numpy())
    return np.concatenate(reps)


def main():
    C.setup_chinese_font()
    device = C.get_device()
    print(f"设备: {device}")
    torch.manual_seed(0)
    np.random.seed(0)

    # 1) 无标注预训练
    pool = build_unlabeled_pool()
    encoder = Encoder(len(C.AAS) + 1)
    print("对比自监督预训练中(InfoNCE, τ=0.1)...")
    pretrain(encoder, pool, device)

    # 2) 标注数据线性探测
    df = C.load_dataset()
    seqs, y = df["seq"].tolist(), df["label"].values
    rep = extract_rep(encoder, seqs, device)
    np.save(os.path.join(C.RESULT_DIR, "contrastive_rep.npy"), rep)
    print(f"预训练表示: {rep.shape}")

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    models = {
        "LR": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0)),
        "SVM": make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0, probability=True)),
    }
    res = C.run_sklearn_models_cv("S11_Contrastive", rep, y, models, feature_names=None)
    for m, (dfo, _oof) in res.items():
        p = dfo[dfo["fold"] == "pooled_OOF"].iloc[0]
        print(f"  [{m}] AUC={p['AUC']:.4f} PR_AUC={p['PR_AUC']:.4f} MCC={p['MCC']:.4f} "
              f"Acc={p['Acc']:.4f} F1={p['F1']:.4f}")


if __name__ == "__main__":
    main()
