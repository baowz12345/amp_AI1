# -*- coding: utf-8 -*-
"""
10_scheme6_tfidf.py   方案6: TF-IDF k-mer 词袋 + SVM/LR
=====================
最简基线(算力极低, 可解释性中等): 3-mer 滑动窗口词袋 -> TF-IDF 加权 -> 线性分类器。
统一协议: StratifiedKFold 10 折 seed=42, 输出 OOF 概率与全套指标(TP/TN/FP/FN/Acc/P/R/F1/MCC/AUC/PR_AUC)。
输出: results/metrics_S12_TFIDF_*.csv, results/oof_S12_TFIDF_*.npy
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def kmers(seq, k=3):
    """3-mer 滑动窗口词袋(位置窗口滑过, 等价于字符 n-gram)"""
    if len(seq) < k:
        return [seq]
    return [seq[i:i + k] for i in range(len(seq) - k + 1)]


def main():
    C.setup_chinese_font()
    df = C.load_dataset()
    seqs, y = df["seq"].tolist(), df["label"].values
    # TF-IDF 词表在每折训练集内拟合 (防数据泄漏), 通过 pipeline 实现
    models = {
        "LR": make_pipeline(TfidfVectorizer(analyzer=kmers, ngram_range=(1, 1), min_df=3,
                                            sublinear_tf=True, dtype=np.float32),
                            StandardScaler(with_mean=False),
                            LogisticRegression(max_iter=3000, C=1.0, solver="lbfgs")),
        "SVM": make_pipeline(TfidfVectorizer(analyzer=kmers, ngram_range=(1, 1), min_df=3,
                                             sublinear_tf=True, dtype=np.float32),
                             StandardScaler(with_mean=False),
                             SVC(kernel="linear", C=1.0, probability=True)),
    }
    res = C.run_sklearn_models_cv("S12_TFIDF", seqs, y, models, feature_names=None)
    for m, (dfo, _oof) in res.items():
        p = dfo[dfo["fold"] == "pooled_OOF"].iloc[0]
        print(f"  [{m}] AUC={p['AUC']:.4f} PR_AUC={p['PR_AUC']:.4f} MCC={p['MCC']:.4f} "
              f"Acc={p['Acc']:.4f} F1={p['F1']:.4f}")


if __name__ == "__main__":
    main()
