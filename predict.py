# -*- coding: utf-8 -*-
"""
predict.py   用训练好的模型对新序列进行分类 (推理脚本)
=================================================================
用法 (命令行):
    python predict.py --input my_peptides.fasta --model S12_TFIDF_LR --out predictions.csv
    python predict.py --input my_peptides.fasta --model S2_XGB --out predictions.csv

输入: FASTA 文件 (每个条目一条肽序列) 或 CSV 文件 (列: seq)
输出: CSV 表格, 每行包含 id / seq / prob_AMP (正类概率) / prediction (AMP / non-AMP)

模型权重位于 models/ 目录 (由 export_models.py 生成):
    models/S2_XGB.joblib       k-mer 组成 + 特征选择 + XGBoost
    models/S12_TFIDF_LR.joblib  3-mer TF-IDF + 逻辑回归
依赖: numpy, pandas, scikit-learn, xgboost, joblib (见 requirements.txt)
"""
import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import common as C  # noqa: E402


def read_fasta(path):
    ids, chunks = [], []
    cur = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().lstrip("\ufeff")
            if not line:
                continue
            if line.startswith(">"):
                ids.append(line[1:].split()[0])
                cur = []
                chunks.append(cur)
            elif cur is None:
                continue  # 文件开头的非序列行, 忽略
            else:
                cur.append(line.upper())
    seqs = ["".join(c) for c in chunks]
    return ids, seqs


def build_kmer_features(seqs):
    Xa, _ = C.aac_features(seqs)
    Xd, _ = C.dpc_features(seqs)
    Xt = C.tripeptide_counts(seqs)
    return np.hstack([Xa, Xd, Xt])


def main():
    ap = argparse.ArgumentParser(description="Classify peptide sequences with a trained AMP model")
    ap.add_argument("--input", required=True, help="Input FASTA file or CSV with a 'seq' column")
    ap.add_argument("--model", choices=["S2_XGB", "S12_TFIDF_LR"], default="S12_TFIDF_LR",
                    help="Model name (default: S12_TFIDF_LR)")
    ap.add_argument("--out", default="predictions.csv", help="Output CSV path")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Probability threshold for the AMP label (default 0.5)")
    a = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(here, "models", a.model + ".joblib")
    if not os.path.exists(model_path):
        sys.exit("Model file not found: %s (run export_models.py first)" % model_path)

    # 读取输入
    if a.input.lower().endswith((".fa", ".fasta", ".faa", ".txt")):
        ids, seqs = read_fasta(a.input)
    else:
        d = pd.read_csv(a.input)
        seqs = d["seq"].astype(str).tolist()
        ids = d.get("id", pd.Series(range(len(seqs)))).tolist()
    print("loaded %d sequences" % len(seqs))

    model = joblib.load(model_path)
    if a.model == "S2_XGB":
        X = build_kmer_features(seqs)
        prob = model.predict_proba(X)[:, 1]
    else:
        prob = model.predict_proba(seqs)[:, 1]

    out = pd.DataFrame({
        "id": ids, "seq": seqs, "prob_AMP": np.round(prob, 6),
        "prediction": np.where(prob >= a.threshold, "AMP", "non-AMP"),
    })
    out.to_csv(a.out, index=False, encoding="utf-8-sig")
    print("saved:", a.out)
    print(out[["id", "prob_AMP", "prediction"]].to_string(index=False))


if __name__ == "__main__":
    main()
