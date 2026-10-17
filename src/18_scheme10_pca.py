# -*- coding: utf-8 -*-
"""
18_scheme10_pca.py   方案10: 降维特征提取(Dimensionality Reduction for Feature Extraction)
==========================================================================================
- 特征: AAC(20) + 理化15 + DPC(400) = 435 维 (稠密, 同方案一体系)
- 降维方法:
    PCA   (StandardScaler -> PCA 保留95%方差)         -> 特征提取/压缩
    t-SNE (perplexity=30, init=pca)                   -> 非线性流形可视化
    UMAP  (n_neighbors=15, min_dist=0.1)              -> 流形学习可视化
- 评估: 原始435维 vs PCA-95%特征 -> LR / XGBoost, 统一10折CV协议(seed=42)
- 信息提取输出:
    results/dimred_summary.csv         (维度/保留率/模型性能 raw vs pca)
    results/figures/fig3_dimreduction_en.png  (PCA方差曲线 + PC1/PC2 + t-SNE + UMAP)
- 结果: results/metrics_S15_raw_*.csv / metrics_S15_PCA_*.csv, oof_S15_*.npy
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import umap
    HAVE_UMAP = True
except Exception:
    HAVE_UMAP = False


def build_features(seqs):
    X_phys, phys_names = C.physico_features(seqs)
    X_aac, aac_names = C.aac_features(seqs)
    X_dpc, dpc_names = C.dpc_features(seqs)
    X = np.hstack([X_phys, X_aac, X_dpc])
    names = phys_names + aac_names + dpc_names
    return X, names


def main():
    df = C.load_dataset()
    seqs, y = df["seq"].tolist(), df["label"].values
    X, names = build_features(seqs)
    print(f"特征矩阵: {X.shape}  (理化15 + AAC20 + DPC400)")

    # ---------- PCA (可视化用全局投影; 评估用每折内拟合防泄漏) ----------
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    pca_vis = PCA(n_components=0.95, random_state=C.CV_SEED).fit(Xs)
    X_pca_vis = pca_vis.transform(Xs)
    cum = np.cumsum(pca_vis.explained_variance_ratio_)
    n95 = pca_vis.n_components_
    print(f"全局 PCA(可视化): {X.shape[1]} -> {n95} 维 (保留95%方差, 累计解释率 {cum[-1]:.4f})")

    # ---------- 分类评估: 原始 435 维 vs PCA(每折训练集内拟合) ----------
    models = {
        "LR": LogisticRegression(max_iter=3000, C=1.0, solver="lbfgs", class_weight="balanced"),
        "XGB": __import__("xgboost").XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                                                   subsample=0.9, colsample_bytree=0.8,
                                                   scale_pos_weight=(1 - y.mean()) / y.mean(),
                                                   n_jobs=-1, random_state=C.CV_SEED,
                                                   eval_metric="logloss"),
    }
    res_raw = C.run_sklearn_models_cv("S15_raw", X, y, models, feature_names=None)
    models_pca = {
        "LR": make_pipeline(StandardScaler(),
                            PCA(n_components=0.95, random_state=C.CV_SEED),
                            LogisticRegression(max_iter=3000, C=1.0, solver="lbfgs",
                                               class_weight="balanced")),
        "XGB": make_pipeline(StandardScaler(),
                             PCA(n_components=0.95, random_state=C.CV_SEED),
                             __import__("xgboost").XGBClassifier(
                                 n_estimators=400, max_depth=6, learning_rate=0.05,
                                 subsample=0.9, colsample_bytree=0.8,
                                 scale_pos_weight=(1 - y.mean()) / y.mean(),
                                 n_jobs=-1, random_state=C.CV_SEED, eval_metric="logloss")),
    }
    res_pca = C.run_sklearn_models_cv("S15_PCA", X, y, models_pca, feature_names=None)

    # ---------- 降维可视化 (全局投影, 仅用于展示) ----------
    rng = np.random.RandomState(C.CV_SEED)
    colors = ["#E8A0A0", "#6BAED6"]  # 负 / 正
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10), dpi=150)

    ax = axes[0, 0]
    ax.plot(np.arange(1, len(cum) + 1), cum, lw=1.8, color="#9BBBF4")
    ax.axhline(0.95, color="k", ls="--", lw=0.9, alpha=0.6)
    ax.axvline(n95, color="gray", ls=":", lw=0.9)
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative explained variance ratio")
    ax.set_title(f"PCA: {X.shape[1]} \u2192 {n95} PCs (95% variance)")
    ax.set_xlim(0, min(len(cum), 150))
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    pc1 = X_pca_vis[:, 0]; pc2 = X_pca_vis[:, 1]
    for lab, c in zip([0, 1], colors):
        idx = y == lab
        ax.scatter(pc1[idx], pc2[idx], s=6, c=c, alpha=0.55,
                   label="Non-animal AMP" if lab == 0 else "Animal AMP")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("First two principal components")
    ax.legend(fontsize=8, markerscale=2)
    ax.grid(alpha=0.2)

    ax = axes[1, 0]
    ts = TSNE(n_components=2, perplexity=30, init="pca", random_state=C.CV_SEED,
              learning_rate="auto").fit_transform(Xs)
    for lab, c in zip([0, 1], colors):
        idx = y == lab
        ax.scatter(ts[idx, 0], ts[idx, 1], s=6, c=c, alpha=0.55,
                   label="Non-animal AMP" if lab == 0 else "Animal AMP")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.set_title("t-SNE projection (perplexity = 30)")
    ax.legend(fontsize=8, markerscale=2)
    ax.grid(alpha=0.2)

    if HAVE_UMAP:
        ax = axes[1, 1]
        red = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=C.CV_SEED,
                        n_jobs=1).fit_transform(Xs)
        for lab, c in zip([0, 1], colors):
            idx = y == lab
            ax.scatter(red[idx, 0], red[idx, 1], s=6, c=c, alpha=0.55,
                       label="Non-animal AMP" if lab == 0 else "Animal AMP")
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        ax.set_title("UMAP projection (n_neighbors = 15)")
        ax.legend(fontsize=8, markerscale=2)
        ax.grid(alpha=0.2)
    else:
        axes[1, 1].axis("off")
        axes[1, 1].text(0.5, 0.5, "UMAP unavailable", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIG_DIR, "fig3_dimreduction_en.png"))
    plt.close(fig)

    # ---------- 汇总表 ----------
    rows = []
    for tag, d in [("raw", res_raw), ("pca", res_pca)]:
        for mname, (dfo, _) in d.items():
            po = dfo[dfo["fold"] == "pooled_OOF"].iloc[0]
            fld = dfo[dfo["fold"] != "pooled_OOF"]
            rows.append({
                "group": tag, "model": po["model"],
                "dims": X.shape[1] if tag == "raw" else n95,
                "Acc": po["Acc"], "Precision": po["Precision"], "Recall": po["Recall"],
                "F1": po["F1"], "MCC": po["MCC"], "AUC": po["AUC"], "PR_AUC": po["PR_AUC"],
                "AUC_mean": fld["AUC"].mean(), "AUC_std": fld["AUC"].std(),
            })
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(C.RESULT_DIR, "dimred_summary.csv"), index=False, encoding="utf-8-sig")
    print("\n降维信息提取汇总 (pooled OOF):")
    print(summ[["model", "dims", "AUC", "PR_AUC", "MCC", "Acc"]].to_string(index=False))
    print(f"PCA 95% 方差 -> {n95} 主成分 (原始 {X.shape[1]} 维), 累计解释率 {cum[-1]:.4f}")


if __name__ == "__main__":
    main()
