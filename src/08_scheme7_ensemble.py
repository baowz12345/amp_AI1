# -*- coding: utf-8 -*-
"""
08_scheme7_ensemble.py
======================
方案七: 模型融合
  (a) 软投票 Soft Voting: 对 7 个基模型(方案一 LR/SVM/RF/XGB + 方案四CNN +
      方案五BiLSTM + 方案六Transformer)的 OOF 概率取平均
  (b) Stacking 堆叠: 以基模型 OOF 概率为元特征, 元学习器用 LR / XGBoost,
      在元特征上做 5 折嵌套交叉验证
基模型 OOF 来自前面各方案脚本保存的 results/oof_*.npy (同一 10 折划分, 无泄漏)。
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import common as C

C.setup_chinese_font()
df = C.load_dataset()
y = df["label"].values

BASE_MODELS = ["S1_LR", "S1_SVM", "S1_RF", "S1_XGB", "S4_CNN", "S5_BiLSTM", "S6_Transformer"]
LABELS = ["LR", "SVM", "RF", "XGBoost", "CNN", "BiLSTM", "Transformer"]

# 加载基模型 OOF
oofs = []
for name in BASE_MODELS:
    p = os.path.join(C.RESULT_DIR, f"oof_{name}.npy")
    if not os.path.exists(p):
        raise FileNotFoundError(f"缺少基模型 OOF: {p}, 请先运行对应方案脚本")
    oofs.append(np.load(p))
OOF = np.column_stack(oofs)   # (N, 7)
print("基模型 OOF 矩阵:", OOF.shape)

# ---------------------------------------------------------------------------
# (a) 软投票
# ---------------------------------------------------------------------------
vote = OOF.mean(axis=1)
m = C.compute_metrics(y, vote)
print(f"\n[软投票] AUC={m['AUC']:.4f} PR_AUC={m['PR_AUC']:.4f} F1={m['F1']:.4f} MCC={m['MCC']:.4f}")
C.save_cv_results("S7_vote", [C.compute_metrics(y, vote)], vote, y, save_oof=True)

# ---------------------------------------------------------------------------
# (b) Stacking (元特征上 5 折嵌套CV)
# ---------------------------------------------------------------------------
def run_stack(meta_name, meta_model):
    from sklearn.model_selection import StratifiedKFold
    skf5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=C.CV_SEED)
    fold_metrics, oof = [], np.zeros(len(y))
    for tr_idx, te_idx in skf5.split(OOF, y):
        mm = meta_model
        from sklearn.base import clone
        mm = clone(meta_model)
        mm.fit(OOF[tr_idx], y[tr_idx])
        prob = mm.predict_proba(OOF[te_idx])[:, 1]
        oof[te_idx] = prob
        fold_metrics.append(C.compute_metrics(y[te_idx], prob))
    full = f"S7_{meta_name}"
    df_folds = pd.DataFrame(fold_metrics)
    df_folds.insert(0, "fold", np.arange(1, 6))
    pooled = C.compute_metrics(y, oof)
    df_out = pd.concat([df_folds, pd.DataFrame([{**pooled, "fold": "pooled_OOF"}])],
                       ignore_index=True)
    df_out.insert(0, "model", full)
    df_out.to_csv(os.path.join(C.RESULT_DIR, f"metrics_{full}.csv"), index=False)
    np.save(os.path.join(C.RESULT_DIR, f"oof_{full}.npy"), oof)
    print(f"[Stacking-{meta_name}] AUC={pooled['AUC']:.4f} PR_AUC={pooled['PR_AUC']:.4f} "
          f"F1={pooled['F1']:.4f} MCC={pooled['MCC']:.4f}")
    return pooled

meta_lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
meta_xgb = __import__("xgboost").XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                               scale_pos_weight=(1 - y.mean()) / y.mean(),
                                               eval_metric="logloss", random_state=C.CV_SEED)
run_stack("stackLR", meta_lr)
run_stack("stackXGB", meta_xgb)
print("方案七完成")
