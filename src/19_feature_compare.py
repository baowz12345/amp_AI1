# -*- coding: utf-8 -*-
"""
19_feature_compare.py   不同类型特征的性能比较 (Feature-Type Benchmark)
=======================================================================
特征组:
  AAC (20) / PHYS (15) / DPC (400) / AAC+PHYS (35) / AAC+DPC (420)
  / PHYS+DPC (415) / ALL (435) / PCA-95% (322)
模型: LR + XGBoost, 统一 10 折 CV 协议 (StratifiedKFold seed=42)
输出:
  results/feature_compare.csv          (每组 pooled + 10 折均值±std)
  results/figures/fig4_feature_compare_en.png  (AUC 分组柱状图)
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FEATURE_GROUPS = {
    "AAC": (["aac"], "AAC (20)"),
    "PHYS": (["phys"], "PHYS (15)"),
    "DPC": (["dpc"], "DPC (400)"),
    "AAC+PHYS": (["aac", "phys"], "AAC+PHYS (35)"),
    "AAC+DPC": (["aac", "dpc"], "AAC+DPC (420)"),
    "PHYS+DPC": (["phys", "dpc"], "PHYS+DPC (415)"),
    "ALL": (["aac", "phys", "dpc"], "ALL (435)"),
    "PCA": (["pca"], "PCA-95% (322)"),
}


def build_all(seqs):
    X_phys, _ = C.physico_features(seqs)
    X_aac, _ = C.aac_features(seqs)
    X_dpc, _ = C.dpc_features(seqs)
    X_all = np.hstack([X_phys, X_aac, X_dpc])
    # 全局 PCA 仅用于确定维数与可视化参考; 评估时的 PCA 在每折训练集内拟合(防泄漏)
    sc = StandardScaler().fit(X_all)
    pca = PCA(n_components=0.95, random_state=C.CV_SEED).fit(sc.transform(X_all))
    return {
        "aac": X_aac, "phys": X_phys, "dpc": X_dpc,
        "aac+phys": np.hstack([X_aac, X_phys]),
        "aac+dpc": np.hstack([X_aac, X_dpc]),
        "phys+dpc": np.hstack([X_phys, X_dpc]),
        "all": X_all,
    }, pca.n_components_


def cv_eval(X, y, model):
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=C.CV_SEED)
    X = np.asarray(X, dtype=np.float64)
    fold_metrics, oof = [], np.zeros(len(y))
    for tr, te in skf.split(X, y):
        m = clone(model)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
        fold_metrics.append(C.compute_metrics(y[te], oof[te]))
    df_folds = pd.DataFrame(fold_metrics)
    pooled = C.compute_metrics(y, oof)
    return df_folds, pooled, oof


def main():
    df = C.load_dataset()
    seqs, y = df["seq"].tolist(), df["label"].values
    feats, n95 = build_all(seqs)
    print(f"特征构建完成; PCA-95% -> {n95} 维")
    model_fns = {
        "LR": lambda: LogisticRegression(max_iter=3000, C=1.0, solver="lbfgs", class_weight="balanced"),
        "XGB": lambda: __import__("xgboost").XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.9,
            colsample_bytree=0.8, scale_pos_weight=(1 - y.mean()) / y.mean(),
            n_jobs=-1, random_state=C.CV_SEED, eval_metric="logloss"),
    }
    rows = []
    for gname, (keys, label) in FEATURE_GROUPS.items():
        if gname == "PCA":
            # PCA 组: 每折训练集内 StandardScaler+PCA(95%) 拟合 (防泄漏)
            Xg = feats["all"]
            for mname, mfn in model_fns.items():
                pipe = make_pipeline(StandardScaler(),
                                     PCA(n_components=0.95, random_state=C.CV_SEED),
                                     mfn())
                df_folds, pooled, oof = cv_eval(Xg, y, pipe)
                rows.append({
                    "group": gname, "label": label, "model": mname,
                    "dims": n95,
                    "Acc": pooled["Acc"], "Precision": pooled["Precision"],
                    "Recall": pooled["Recall"], "F1": pooled["F1"], "MCC": pooled["MCC"],
                    "AUC": pooled["AUC"], "PR_AUC": pooled["PR_AUC"],
                    "AUC_mean": df_folds["AUC"].mean(), "AUC_std": df_folds["AUC"].std(),
                    "MCC_mean": df_folds["MCC"].mean(), "MCC_std": df_folds["MCC"].std(),
                    "F1_mean": df_folds["F1"].mean(), "F1_std": df_folds["F1"].std(),
                })
            continue
        Xg = feats[gname.lower()]
        for mname, mfn in model_fns.items():
            df_folds, pooled, oof = cv_eval(Xg, y, mfn())
            rows.append({
                "group": gname, "label": label, "model": mname,
                "dims": Xg.shape[1],
                "Acc": pooled["Acc"], "Precision": pooled["Precision"],
                "Recall": pooled["Recall"], "F1": pooled["F1"], "MCC": pooled["MCC"],
                "AUC": pooled["AUC"], "PR_AUC": pooled["PR_AUC"],
                "AUC_mean": df_folds["AUC"].mean(), "AUC_std": df_folds["AUC"].std(),
                "MCC_mean": df_folds["MCC"].mean(), "MCC_std": df_folds["MCC"].std(),
                "F1_mean": df_folds["F1"].mean(), "F1_std": df_folds["F1"].std(),
            })
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(C.RESULT_DIR, "feature_compare.csv"), index=False, encoding="utf-8-sig")
    print(summ[["group", "model", "dims", "AUC", "AUC_mean", "AUC_std", "MCC", "Acc", "F1"]]
          .to_string(index=False))

    # 柱状图: 各组 AUC (LR vs XGB), 按 XGB AUC 降序
    xgb = summ[summ["model"] == "XGB"].set_index("group")
    lr = summ[summ["model"] == "LR"].set_index("group")
    order = xgb.sort_values("AUC_mean", ascending=False).index.tolist()
    x = np.arange(len(order))
    w = 0.36
    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=150)
    ax.bar(x - w / 2, [xgb.loc[g, "AUC_mean"] for g in order], w, yerr=[xgb.loc[g, "AUC_std"] for g in order],
           label="LR", color="#9BBBF4", capsize=3, error_kw={"lw": 1})
    ax.bar(x + w / 2, [lr.loc[g, "AUC_mean"] for g in order], w, yerr=[lr.loc[g, "AUC_std"] for g in order],
           label="XGBoost", color="#94D8C3", capsize=3, error_kw={"lw": 1})
    ax.set_xticks(x)
    ax.set_xticklabels([FEATURE_GROUPS[g][1] for g in order], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("AUC (10-fold CV mean \u00b1 SD)")
    ax.set_title("Performance comparison across feature types (10-fold CV)")
    ax.legend()
    ax.set_ylim(0.7, 1.0)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIG_DIR, "fig4_feature_compare_en.png"))
    plt.close(fig)
    print("已输出: feature_compare.csv + fig4_feature_compare_en.png")


if __name__ == "__main__":
    main()
