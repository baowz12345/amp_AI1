# -*- coding: utf-8 -*-
"""
02_scheme1_physicochem.py
=========================
方案一: 物理化学描述符 + 氨基酸组成 + 传统机器学习
  特征: peptides库15维物理化学(分子量/电荷/pI/疏水性/疏水矩/脂肪指数/Boman/不稳定指数/
        最长同氨基酸串/z-scales×5) + AAC 20维 = 35维
  模型: LogisticRegression, SVM(RBF), RandomForest, XGBoost
  评估: 10折分层交叉验证, 输出 TP/TN/FP/FN/Acc/P/R/F1/MCC/AUC/PR_AUC + 特征重要性
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import common as C

C.setup_chinese_font()
df = C.load_dataset()
seqs = df["seq"].tolist()
y = df["label"].values

print("== 构建物理化学特征 ==")
X_phys, phys_names = C.physico_features(seqs)
X_aac, aac_names = C.aac_features(seqs)
X = np.hstack([X_phys, X_aac])
names = phys_names + aac_names
print(f"特征矩阵: {X.shape}  (物理化学15 + AAC20)")

models = {
    "LR": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
    "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale",
                                               class_weight="balanced", probability=True,
                                               random_state=C.CV_SEED)),
    "RF": RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=3,
                                 class_weight="balanced", n_jobs=-1, random_state=C.CV_SEED),
    "XGB": __import__("xgboost").XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                                               subsample=0.9, colsample_bytree=0.8,
                                               scale_pos_weight=(1 - y.mean()) / y.mean(),
                                               n_jobs=-1, random_state=C.CV_SEED,
                                               eval_metric="logloss"),
}

print("== 10折交叉验证 ==")
res = C.run_sklearn_models_cv("S1", X, y, models, feature_names=names, topk_importance=30)

summary = []
for m, (df_folds, oof) in res.items():
    line = {"方案": "S1_物理化学", "模型": m}
    line.update(C.format_metric(df_folds[df_folds["fold"] != "pooled_OOF"]))
    summary.append(line)
    print(f"  [{m}] AUC={line['AUC']}  F1={line['F1']}  MCC={line['MCC']}")

pd.DataFrame(summary).to_csv(os.path.join(C.RESULT_DIR, "summary_S1.csv"), index=False, encoding="utf-8-sig")
print("方案一完成, 结果见 results/metrics_S1_*.csv 与 importance_S1_*.csv")
