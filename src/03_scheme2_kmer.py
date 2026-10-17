# -*- coding: utf-8 -*-
"""
03_scheme2_kmer.py
==================
方案二: k-mer 组成特征 + 特征选择 + 经典分类器
  特征: AAC(20) + DPC(400) + 三肽组成(8000) = 8420 维, 经方差过滤 + 互信息(MI)选 top-500
  模型: GaussianNB, KNN, LogisticRegression, RandomForest, XGBoost
  评估: 10折分层交叉验证 + 特征重要性
"""
import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import common as C

C.setup_chinese_font()
df = C.load_dataset()
seqs = df["seq"].tolist()
y = df["label"].values

print("== 构建 k-mer 组成特征 (AAC+DPC+三肽) ==")
X_aac, aac_names = C.aac_features(seqs)
X_dpc, dpc_names = C.dpc_features(seqs)
X_tri = C.tripeptide_counts(seqs)
X_raw = np.hstack([X_aac, X_dpc, X_tri])
raw_names = aac_names + dpc_names + [f"TRP_{a}{b}{c}" for a in C.AAS for b in C.AAS for c in C.AAS]
print(f"原始特征: {X_raw.shape}")

# 特征选择移入每折 pipeline (防数据泄漏): 方差过滤 + 互信息 top-500 仅在训练折内拟合
FEATSEL = [VarianceThreshold(threshold=1e-5),
           SelectKBest(mutual_info_classif, k=500)]
models = {
    "NB": make_pipeline(*FEATSEL, GaussianNB()),
    "KNN": make_pipeline(*FEATSEL, StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
    "LR": make_pipeline(*FEATSEL, StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
    "RF": make_pipeline(*FEATSEL, RandomForestClassifier(n_estimators=300, max_depth=14, min_samples_leaf=2,
                                                        class_weight="balanced", n_jobs=-1, random_state=C.CV_SEED)),
    "XGB": make_pipeline(*FEATSEL, __import__("xgboost").XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                                                                       subsample=0.9, colsample_bytree=0.8,
                                                                       scale_pos_weight=(1 - y.mean()) / y.mean(),
                                                                       n_jobs=-1, random_state=C.CV_SEED,
                                                                       eval_metric="logloss")),
}

print("== 10折交叉验证 (特征选择在每折训练集内拟合, 无泄漏) ==")
res = C.run_sklearn_models_cv("S2", X_raw, y, models, feature_names=None, topk_importance=30)
summary = []
for m, (df_folds, oof) in res.items():
    line = {"方案": "S2_kmer组成", "模型": m}
    line.update(C.format_metric(df_folds[df_folds["fold"] != "pooled_OOF"]))
    summary.append(line)
    print(f"  [{m}] AUC={line['AUC']}  F1={line['F1']}  MCC={line['MCC']}")
pd.DataFrame(summary).to_csv(os.path.join(C.RESULT_DIR, "summary_S2.csv"), index=False, encoding="utf-8-sig")
print("方案二完成")
