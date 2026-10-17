# -*- coding: utf-8 -*-
"""
21_case_study.py   Case Study: 端到端案例研究 (选定真实肽序列走全流程)
=====================================================================
- 案例: LL-37 (人 cathelicidin) / Magainin-2 (非洲爪蟾) / Cecropin-A (蛾) [动物源=正例]
        + 植物源防御素样肽 [非动物源=负例对照]
- 流程: 435维特征(AAC+理化+DPC) -> 10折CV内每折训练 -> 对案例预测(均值±std)
        模型: XGB-435 / LR-435 / XGB-PCA(95%) / LR-PCA / TFIDF-LR(3-mer)
        另输出: PCA投影位置图、XGB特征归因(top gain 特征 z 值)
- 产物:
    results/case_study_pred.csv   (case x model 预测概率 mean/std)
    results/case_study_attr.csv   (案例属性 + top 判别特征值)
    results/figures/fig5_case_study_pred.png
    results/figures/fig6_case_study_pca.png
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
import xgboost
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CASES = [
    ("LL-37", "Human cathelicidin LL-37",
     "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES", 1),
    ("Magainin-2", "Xenopus (African clawed frog) magainin-2",
     "GIGKFLHSAKKFGKAFVGEIMNS", 1),
    ("Cecropin-A", "Moth (Hyalophora cecropia) cecropin-A",
     "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK", 1),
    ("Defensin-like", "Plant defensin-like peptide (non-animal control)",
     "GWFTAVQLTLAGRCGRWFTGSFECTTNNVKCG", 0),
]
MODELS = ["XGB-435", "LR-435", "XGB-PCA", "LR-PCA", "TFIDF-LR"]


def kmers(seq, k=3):
    if len(seq) < k:
        return [seq]
    return [seq[i:i + k] for i in range(len(seq) - k + 1)]


def build_features(seqs):
    X_phys, pn = C.physico_features(seqs)
    X_aac, an = C.aac_features(seqs)
    X_dpc, dn = C.dpc_features(seqs)
    return np.hstack([X_phys, X_aac, X_dpc]), pn + an + dn


def main():
    df = C.load_dataset()
    seqs_all, y_all = df["seq"].tolist(), df["label"].values
    X_all, names = build_features(seqs_all)
    print(f"特征矩阵: {X_all.shape}")

    case_seqs = [c[2] for c in CASES]
    X_case, _ = build_features(case_seqs)
    X_all_plus = np.vstack([X_all, X_case])

    # ---------- 10 折: 每折训练 -> 案例预测 ----------
    skf = C.stratified_kfold()
    folds = list(skf.split(X_all, y_all))
    preds = {m: np.zeros((len(CASES), len(folds))) for m in MODELS}
    pca_dims = []
    for fi, (tr, te) in enumerate(folds):
        Xtr, ytr = X_all[tr], y_all[tr]
        # 435 维模型
        sc = StandardScaler().fit(Xtr)
        xgb = xgboost.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                                    subsample=0.9, colsample_bytree=0.8,
                                    scale_pos_weight=(1 - ytr.mean()) / ytr.mean(),
                                    n_jobs=-1, random_state=C.CV_SEED, eval_metric="logloss")
        xgb.fit(sc.transform(Xtr), ytr)
        preds["XGB-435"][:, fi] = xgb.predict_proba(sc.transform(X_case))[:, 1]
        lr = LogisticRegression(max_iter=3000, C=1.0, solver="lbfgs", class_weight="balanced")
        lr.fit(sc.transform(Xtr), ytr)
        preds["LR-435"][:, fi] = lr.predict_proba(sc.transform(X_case))[:, 1]
        # PCA 变体 (95% 方差, 每折独立)
        pca = PCA(n_components=0.95, random_state=C.CV_SEED).fit(sc.transform(Xtr))
        pca_dims.append(pca.n_components_)
        xgbp = xgboost.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                                     subsample=0.9, colsample_bytree=0.8,
                                     scale_pos_weight=(1 - ytr.mean()) / ytr.mean(),
                                     n_jobs=-1, random_state=C.CV_SEED, eval_metric="logloss")
        xgbp.fit(pca.transform(sc.transform(Xtr)), ytr)
        preds["XGB-PCA"][:, fi] = xgbp.predict_proba(pca.transform(sc.transform(X_case)))[:, 1]
        lrp = LogisticRegression(max_iter=3000, C=1.0, solver="lbfgs", class_weight="balanced")
        lrp.fit(pca.transform(sc.transform(Xtr)), ytr)
        preds["LR-PCA"][:, fi] = lrp.predict_proba(pca.transform(sc.transform(X_case)))[:, 1]
        # TF-IDF 3-mer
        vec = TfidfVectorizer(analyzer=kmers, ngram_range=(1, 1), min_df=3,
                              sublinear_tf=True, dtype=np.float32)
        Xtf = vec.fit_transform([seqs_all[i] for i in tr])
        Xtf_case = vec.transform(case_seqs)
        lrt = make_pipeline(StandardScaler(with_mean=False),
                            LogisticRegression(max_iter=3000, C=1.0, solver="lbfgs"))
        lrt.fit(Xtf, ytr)
        preds["TFIDF-LR"][:, fi] = lrt.predict_proba(Xtf_case)[:, 1]
        print(f"fold {fi + 1}/{len(folds)} done (PCA dims = {pca.n_components_})")

    # ---------- 预测汇总表 ----------
    rows = []
    for ci, (name, src, seq, lab) in enumerate(CASES):
        for m in MODELS:
            p = preds[m][ci]
            rows.append({
                "case": name, "label": "Animal AMP" if lab == 1 else "Non-animal control",
                "model": m,
                "p_mean": float(p.mean()), "p_std": float(p.std()),
                "p_min": float(p.min()), "p_max": float(p.max()),
            })
    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(os.path.join(C.RESULT_DIR, "case_study_pred.csv"),
                   index=False, encoding="utf-8-sig")
    print("\nCase predictions (10-fold mean \u00b1 SD):")
    print(pred_df.pivot(index="case", columns="model", values="p_mean").round(4).to_string())
    print("PCA dims per fold:", pca_dims, "mean =", np.mean(pca_dims).round(1))

    # ---------- 特征归因 (全数据 XGB gain importance) ----------
    scf = StandardScaler().fit(X_all)
    xgb_full = xgboost.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                                     subsample=0.9, colsample_bytree=0.8,
                                     scale_pos_weight=(1 - y_all.mean()) / y_all.mean(),
                                     n_jobs=-1, random_state=C.CV_SEED, eval_metric="logloss")
    xgb_full.fit(scf.transform(X_all), y_all)
    imp = pd.Series(xgb_full.feature_importances_, index=names).sort_values(ascending=False)
    # 展示用特征: 可解释的原始生物学属性 (理化 + 组成), 取案例间 z 区分度最高的组合
    cand = ["pI", "charge_pH7", "hydrophobicity", "hydrophobic_moment", "aliphatic_index",
            "AAC_Y", "AAC_F", "AAC_W", "AAC_R", "AAC_K",
            "DPC_IE", "DPC_MA", "DPC_ST", "DPC_CV", "DPC_EM"]
    zc = scf.transform(X_case)
    zvar = {f: float(np.std([zc[i, names.index(f)] for i in range(len(CASES))]))
            for f in cand if f in names}
    sel4 = sorted(zvar, key=lambda f: -zvar[f])[:4]
    attr_rows = []
    for ci, (name, src, seq, lab) in enumerate(CASES):
        attr_rows.append({"case": name, "label": "Animal AMP" if lab == 1 else "Non-animal control",
                          "seq": seq, "len": len(seq), "source": src})
        for feat in sel4:
            attr_rows[-1][feat] = f"{X_case[ci, names.index(feat)]:.3f}"
    attr_df = pd.DataFrame(attr_rows)
    attr_df.to_csv(os.path.join(C.RESULT_DIR, "case_study_attr.csv"),
                   index=False, encoding="utf-8-sig")
    pd.Series(sel4, name="sel_feat").to_csv(
        os.path.join(C.RESULT_DIR, "case_study_topfeat.csv"), index=False, encoding="utf-8-sig")
    print("\nSelected interpretable features (z-dispersive):", sel4)
    print("XGB gain of selected:", [round(float(imp[f]), 4) if f in imp.index else None for f in sel4])
    print("\nTop-8 discriminative features (XGB gain):")
    print(imp.head(8).to_string())
    print("\nattribute table columns:", list(attr_df.columns))

    # ---------- 图 5: 案例预测概率 (mean ± SD) ----------
    colors_m = ["#9BBBF4", "#A3D5E8", "#E1B98F", "#94D8C3", "#C9A7E8"]
    fig, ax = plt.subplots(figsize=(11.5, 5.6), dpi=150)
    ncase, nm = len(CASES), len(MODELS)
    x = np.arange(ncase)
    w = 0.16
    for mi, m in enumerate(MODELS):
        pv = pred_df[pred_df["model"] == m]
        ax.bar(x + (mi - (nm - 1) / 2) * w, pv["p_mean"], w, yerr=pv["p_std"],
               label=m, color=colors_m[mi], capsize=3, error_kw={"lw": 1.0})
    ax.axhline(0.5, color="k", ls="--", lw=1.0, alpha=0.7)
    ax.text(-0.45, 0.505, "0.5 threshold", fontsize=9)
    ax.set_xticks(x)
    labels = [f"{n}\n({'Animal' if l == 1 else 'Non-animal'})" for n, _, _, l in CASES]
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Predicted P(animal AMP), 10-fold mean \u00b1 SD")
    ax.set_title("Case-study predictions across representative models")
    ax.legend(ncol=5, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, 1.02),
              frameon=False, handlelength=1.2, columnspacing=1.0)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIG_DIR, "fig5_case_study_pred.png"))
    plt.close(fig)

    # ---------- 图 6: 案例在 PCA 空间位置 ----------
    scg = StandardScaler().fit(X_all)
    pcag = PCA(n_components=0.95, random_state=C.CV_SEED).fit(scg.transform(X_all))
    proj_all = pcag.transform(scg.transform(X_all))
    proj_case = pcag.transform(scg.transform(X_case))
    colors = ["#E8A0A0", "#6BAED6"]
    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=150)
    for lab, c in zip([0, 1], colors):
        idx = y_all == lab
        ax.scatter(proj_all[idx, 0], proj_all[idx, 1], s=7, c=c, alpha=0.5,
                   label="Non-animal AMP" if lab == 0 else "Animal AMP")
    for ci, (name, _, _, lab) in enumerate(CASES):
        ax.scatter(proj_case[ci, 0], proj_case[ci, 1], marker="*", s=230, c="k",
                   edgecolors="w", linewidths=0.8, zorder=5)
        ax.annotate(name, (proj_case[ci, 0], proj_case[ci, 1]),
                    textcoords="offset points", xytext=(10, 8), fontsize=9, fontweight="bold")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("Case-study peptides in the PCA projection (322 PCs, 95% variance)")
    ax.legend(fontsize=9, markerscale=1.5)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIG_DIR, "fig6_case_study_pca.png"))
    plt.close(fig)

    print("\nfigures saved: fig5_case_study_pred.png / fig6_case_study_pca.png")


if __name__ == "__main__":
    main()
