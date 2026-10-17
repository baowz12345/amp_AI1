# -*- coding: utf-8 -*-
"""
22_feature_fusion.py   多特征融合实验 (Multi-Feature Fusion)
=====================================================================
- 单特征集: AAC(20) / PHYS(15) / DPC(400)  (同一特征体系)
- 特征级融合 (early fusion): AAC+PHYS+DPC 拼接 = 435 维 (即方案10 raw 特征)
- 决策级融合 (late fusion): 三个单特征模型的预测概率等权平均 (10 折一致协议)
- 模型: XGB / LR (与全篇统一超参, StratifiedKFold 10 折 seed=42)
- 输出:
    results/feature_fusion.csv      (模型 x mean±std AUC/MCC/F1/Acc + pooled AUC)
    results/figures/fig11_feature_fusion_en.png  (early vs late vs single 对比)
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from sklearn.linear_model import LogisticRegression
import xgboost
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SINGLES = {"AAC": 0, "PHYS": 1, "DPC": 2}


def build_single(seqs):
    Xp, _ = C.physico_features(seqs)
    Xa, _ = C.aac_features(seqs)
    Xd, _ = C.dpc_features(seqs)
    return {"AAC": Xa, "PHYS": Xp, "DPC": Xd}


def make_xgb(ytr):
    return xgboost.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                                 subsample=0.9, colsample_bytree=0.8,
                                 scale_pos_weight=(1 - ytr.mean()) / ytr.mean(),
                                 n_jobs=-1, random_state=C.CV_SEED, eval_metric="logloss")


def make_lr():
    return LogisticRegression(max_iter=3000, C=1.0, solver="lbfgs", class_weight="balanced")


def fold_metrics(y, p):
    return C.compute_metrics(y, p)


def main():
    df = C.load_dataset()
    seqs, y = df["seq"].tolist(), df["label"].values
    Xs = build_single(seqs)
    X_early = np.hstack([Xs["PHYS"], Xs["AAC"], Xs["DPC"]])   # 与方案10一致顺序
    skf = C.stratified_kfold()
    folds = list(skf.split(X_early, y))

    models = {
        "AAC-XGB": ("XGB", "AAC"), "PHYS-XGB": ("XGB", "PHYS"), "DPC-XGB": ("XGB", "DPC"),
        "AAC-LR": ("LR", "AAC"), "PHYS-LR": ("LR", "PHYS"), "DPC-LR": ("LR", "DPC"),
        "EARLY-XGB": ("XGB", None, "early"), "EARLY-LR": ("LR", None, "early"),
        "LATE-XGB": (None, None, "late_xgb"), "LATE-LR": (None, None, "late_lr"),
    }
    # 每折记录: {model: (p_test, y_test)}
    per_fold = {m: [] for m in models}
    for fi, (tr, te) in enumerate(folds):
        ytr, yte = y[tr], y[te]
        prob = {}
        for tag in ["AAC", "PHYS", "DPC"]:
            Xtr, Xte = Xs[tag][tr], Xs[tag][te]
            for kind in ["XGB", "LR"]:
                m = make_xgb(ytr) if kind == "XGB" else make_lr()
                m.fit(Xtr, ytr)
                prob[f"{tag}-{kind}"] = m.predict_proba(Xte)[:, 1]
        Xtr_e, Xte_e = X_early[tr], X_early[te]
        for kind in ["XGB", "LR"]:
            m = make_xgb(ytr) if kind == "XGB" else make_lr()
            m.fit(Xtr_e, ytr)
            prob[f"EARLY-{kind}"] = m.predict_proba(Xte_e)[:, 1]
        prob["LATE-XGB"] = (prob["AAC-XGB"] + prob["PHYS-XGB"] + prob["DPC-XGB"]) / 3
        prob["LATE-LR"] = (prob["AAC-LR"] + prob["PHYS-LR"] + prob["DPC-LR"]) / 3
        for mname, p in prob.items():
            per_fold[mname].append((p, yte))
        print(f"fold {fi + 1}/{len(folds)} done")
    _ = models  # noqa

    # ---------- 汇总 (mean±std + pooled) ----------
    rows = []
    for mname in models:
        per = []
        all_p, all_y = [], []
        for p, yt in per_fold[mname]:
            all_p.append(p); all_y.append(yt)
            per.append(C.compute_metrics(yt, p))
        pooled = C.compute_metrics(np.concatenate(all_y), np.concatenate(all_p))
        rows.append({
            "model": mname,
            "Acc_mean": np.mean([m["Acc"] for m in per]), "Acc_std": np.std([m["Acc"] for m in per]),
            "F1_mean": np.mean([m["F1"] for m in per]), "F1_std": np.std([m["F1"] for m in per]),
            "MCC_mean": np.mean([m["MCC"] for m in per]), "MCC_std": np.std([m["MCC"] for m in per]),
            "AUC_mean": np.mean([m["AUC"] for m in per]), "AUC_std": np.std([m["AUC"] for m in per]),
            "PR_AUC_mean": np.mean([m["PR_AUC"] for m in per]), "PR_AUC_std": np.std([m["PR_AUC"] for m in per]),
            "AUC_pooled": pooled["AUC"], "PR_AUC_pooled": pooled["PR_AUC"], "MCC_pooled": pooled["MCC"],
        })
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(C.RESULT_DIR, "feature_fusion.csv"), index=False, encoding="utf-8-sig")
    print("\nMulti-feature fusion summary (10-fold mean \u00b1 SD):")
    print(summ[["model", "AUC_mean", "AUC_std", "MCC_mean", "MCC_std", "AUC_pooled"]].to_string(index=False))

    # ---------- 图: early vs late vs single ----------
    order = ["AAC-XGB", "PHYS-XGB", "DPC-XGB", "EARLY-XGB", "LATE-XGB",
             "AAC-LR", "PHYS-LR", "DPC-LR", "EARLY-LR", "LATE-LR"]
    d = summ.set_index("model").loc[order]
    colors = {"XGB": "#9BBBF4", "LR": "#A3D5E8"}
    fig, ax = plt.subplots(figsize=(11.5, 5.4), dpi=150)
    x = np.arange(len(order))
    bars = ax.bar(x, d["AUC_mean"], 0.6, yerr=d["AUC_std"],
                  color=[colors[m.split("-")[-1]] for m in order], capsize=3,
                  error_kw={"lw": 0.9})
    ax.axhline(0.95, color="k", ls="--", lw=1.0, alpha=0.6)
    ax.text(-0.4, 0.952, "0.95", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("EARLY", "Early").replace("LATE", "Late").replace("-", "\n")
                        for m in order], fontsize=8)
    ax.set_ylim(0.78, 1.0)
    ax.set_ylabel("AUC (10-fold CV mean \u00b1 SD)")
    ax.set_title("Multi-feature fusion: single vs. early (concat 435-d) vs. late (probability averaging)")
    ax.grid(axis="y", alpha=0.3)
    # 融合标记
    for i, m in enumerate(order):
        if m.startswith("EARLY") or m.startswith("LATE"):
            bars[i].set_edgecolor("#1A1B1C"); bars[i].set_linewidth(1.5)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc=colors["XGB"], label="XGBoost"),
                       Patch(fc=colors["LR"], label="Logistic regression"),
                       Patch(fc="none", ec="#1A1B1C", lw=1.5, label="Fusion (early/late)")],
              fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIG_DIR, "fig11_feature_fusion_en.png"))
    plt.close(fig)
    print("\nfig saved: fig11_feature_fusion_en.png")


if __name__ == "__main__":
    main()
