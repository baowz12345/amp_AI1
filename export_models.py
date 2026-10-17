# -*- coding: utf-8 -*-
"""
export_models.py   导出部署模型权重 (编辑意见: 提供模型本身)
=================================================================
为便于药物化学家直接使用, 用完整数据集重新训练并导出两个代表性的
可直接推理的模型 (joblib 格式):
  - models/S2_XGB.joblib      : k-mer 组成(8420) -> 方差过滤+MI top-500 -> XGBoost
  - models/S12_TFIDF_LR.joblib : 3-mer TF-IDF -> 标准化 -> 逻辑回归
部署模型的性能由论文中的 10 折交叉验证结果给出; 此处用全数据训练
仅用于提供开箱即用的权重, 与论文评估协议无关。
用法: python export_models.py
"""
import os
import sys
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "amp_github_release", "models")
os.makedirs(OUT, exist_ok=True)


def main():
    df = C.load_dataset()
    seqs = df["seq"].tolist()
    y = df["label"].values
    print("样本数:", len(y), " 正类:", int(y.sum()))

    # ---- S2_XGB: k-mer 组成特征 ----
    X_aac, _ = C.aac_features(seqs)
    X_dpc, _ = C.dpc_features(seqs)
    X_tri = C.tripeptide_counts(seqs)
    X_raw = np.hstack([X_aac, X_dpc, X_tri])
    print("S2_XGB 特征矩阵:", X_raw.shape)
    pipe_s2 = make_pipeline(
        VarianceThreshold(threshold=1e-5),
        SelectKBest(mutual_info_classif, k=500),
        StandardScaler(),
        xgboost.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                              subsample=0.9, colsample_bytree=0.8,
                              scale_pos_weight=(1 - y.mean()) / y.mean(),
                              n_jobs=-1, random_state=C.CV_SEED, eval_metric="logloss"),
    )
    pipe_s2.fit(X_raw, y)
    joblib.dump(pipe_s2, os.path.join(OUT, "S2_XGB.joblib"), compress=3)
    print("saved:", os.path.join(OUT, "S2_XGB.joblib"))

    # ---- S12_TFIDF_LR: 3-mer TF-IDF ----
    pipe_tfidf = make_pipeline(
        TfidfVectorizer(analyzer=C.kmers3, ngram_range=(1, 1), min_df=3,
                        sublinear_tf=True, dtype=np.float32),
        StandardScaler(with_mean=False),
        LogisticRegression(max_iter=3000, C=1.0, solver="lbfgs"),
    )
    pipe_tfidf.fit(seqs, y)
    joblib.dump(pipe_tfidf, os.path.join(OUT, "S12_TFIDF_LR.joblib"), compress=3)
    print("saved:", os.path.join(OUT, "S12_TFIDF_LR.joblib"))

    # 自检: 训练集回代概率范围
    p1 = pipe_s2.predict_proba(X_raw)[:, 1]
    p2 = pipe_tfidf.predict_proba(seqs)[:, 1]
    print("S2_XGB train prob mean %.3f; S12_TFIDF_LR train prob mean %.3f" % (p1.mean(), p2.mean()))


if __name__ == "__main__":
    main()
