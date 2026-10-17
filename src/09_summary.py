# -*- coding: utf-8 -*-
"""
09_summary.py
=============
汇总所有方案结果:
  1) 构建总对比表(10折 mean±std + OOF 混淆矩阵 TP/TN/FP/FN)
  2) 汇总绘制 ROC 曲线、PR 曲线 (所有模型)
  3) 性能柱状对比图 (AUC/F1/MCC)
  4) 特征重要性 top20 (方案一 RF/XGB、方案二 XGB)
  5) 实验流程图(方案架构总览)
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import common as C

C.setup_chinese_font()

df = C.load_dataset()
y = df["label"].values
OOF_GLOB = os.path.join(C.RESULT_DIR, "oof_*.npy")
metrics_files = sorted(glob.glob(os.path.join(C.RESULT_DIR, "metrics_*.csv")))


def load_pooled_table():
    rows = []
    rate_cols = ["Acc", "Precision", "Recall", "F1", "MCC", "AUC", "PR_AUC"]
    scheme_map = {"S1": "方案一 物理化学+ML", "S2": "方案二 k-mer+特征选择", "S3": "方案三 Word2Vec",
                  "S4": "方案四 CNN", "S5": "方案五 BiLSTM+Attention", "S6": "方案六 Transformer",
                  "S7": "方案七 集成融合"}
    for fp in metrics_files:
        mdf = pd.read_csv(fp)
        pooled = mdf[mdf["fold"] == "pooled_OOF"]
        folds = mdf[mdf["fold"] != "pooled_OOF"]
        if pooled.empty or folds.empty:
            continue
        name = mdf["model"].iloc[0]
        prefix = name.split("_")[0]
        row = {"方案": scheme_map.get(prefix, prefix), "模型": name}
        if len(folds) == 1:                      # 无折间结构(如软投票) -> 整体值
            for k in rate_cols:
                row[k] = f"{pooled.iloc[0][k]:.4f}"
        else:
            fm = C.format_metric(folds)
            row.update({k: fm[k] for k in rate_cols if k in fm})
        for k in ("TP", "FP", "TN", "FN"):
            row[f"OOF_{k}"] = int(pooled.iloc[0][k])
        rows.append(row)
    return pd.DataFrame(rows)


def plot_roc_pr():
    oof_files = sorted(glob.glob(OOF_GLOB))
    plt.figure(figsize=(12, 5))
    # ROC
    ax = plt.subplot(1, 2, 1)
    for fp in oof_files:
        name = os.path.basename(fp)[4:-4]
        prob = np.load(fp)
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y, prob)
        ax.plot(fpr, tpr, lw=1.6, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC 曲线 (10折OOF)")
    ax.legend(fontsize=7, loc="lower right", ncol=2)
    # PR
    ax2 = plt.subplot(1, 2, 2)
    for fp in oof_files:
        name = os.path.basename(fp)[4:-4]
        prob = np.load(fp)
        from sklearn.metrics import precision_recall_curve, average_precision_score
        prec, rec, _ = precision_recall_curve(y, prob)
        ax2.plot(rec, prec, lw=1.6, label=f"{name} (PR={average_precision_score(y, prob):.3f})")
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
    ax2.set_title("PR 曲线 (10折OOF)")
    ax2.legend(fontsize=7, loc="lower left", ncol=2)
    plt.tight_layout()
    out = os.path.join(C.FIG_DIR, "roc_pr_all.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print("ROC/PR 汇总图:", out)


def plot_bars(tab):
    """按 AUC 排序的柱状图, 只画 10 折有 std 的行(软投票单独标注)"""
    tab = tab.copy()
    tab["AUC_mean"] = [float(v.split("±")[0]) for v in tab["AUC"]]
    tab["AUC_std"] = [float(v.split("±")[1]) if "±" in v else 0 for v in tab["AUC"]]
    tab = tab.sort_values("AUC_mean", ascending=False)
    names = tab["模型"].tolist()
    x = np.arange(len(names))
    w = 0.27
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for i, (col, color) in enumerate([("AUC", "#1f77b4"), ("F1", "#ff7f0e"), ("MCC", "#2ca02c")]):
        vals = [float(v.split("±")[0]) for v in tab[col]]
        ax.bar(x + (i - 1) * w, vals, w, label=col, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("得分")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(C.FIG_DIR, "performance_bar.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print("性能柱状图:", out)


def plot_importance():
    files = [os.path.join(C.RESULT_DIR, "importance_S1_RF.csv"),
             os.path.join(C.RESULT_DIR, "importance_S1_XGB.csv"),
             os.path.join(C.RESULT_DIR, "importance_S2_XGB.csv")]
    files = [f for f in files if os.path.exists(f)]
    if not files:
        print("无特征重要性文件, 跳过")
        return
    n = len(files)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5))
    for ax, fp in zip(axes, files):
        d = pd.read_csv(fp).head(20).iloc[::-1]
        ax.barh(d["feature"], d["importance"], color="#9467bd")
        ax.set_title(os.path.basename(fp).replace("importance_", "").replace(".csv", ""))
        ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    out = os.path.join(C.FIG_DIR, "feature_importance.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print("特征重要性图:", out)


def plot_workflow():
    """实验流程图: 数据 -> 特征/表示 -> 模型 -> 评估"""
    fig, ax = plt.subplots(figsize=(12, 6.6))
    ax.axis("off")
    boxes = [
        (0.5, 0.93, "APD2024a 数据\nanimalAMP 2580 / naturalAMP 3306", "#fde0dc", 0.16),
        (0.5, 0.78, "预处理 & 任务定义\n清洗(标准AA, 4-100aa, 去重)\n正类=动物源2492 / 负类=非动物源716", "#d9e8fc", 0.15),
    ]
    # 四条特征/表示路径
    paths = [
        (0.13, 0.56, "S1 物理化学特征\n15维描述符 + AAC20", "#fff3bf"),
        (0.38, 0.56, "S2 k-mer组成\nAAC+DPC+三肽 8420维\n方差+MI选500", "#fff3bf"),
        (0.63, 0.56, "S3 Word2Vec\n3-mer词向量 64维\n(每折重训)", "#fff3bf"),
        (0.88, 0.56, "S4/S5/S6 深度学习\nOne-hot→CNN | 索引→\nBiLSTM+Attn / Transformer", "#ffe8cc"),
    ]
    models = [
        (0.13, 0.33, "LR / SVM / RF / XGBoost", "#e7f5dd"),
        (0.38, 0.33, "NB / KNN / LR / RF / XGBoost", "#e7f5dd"),
        (0.63, 0.33, "LR / SVM / RF", "#e7f5dd"),
        (0.88, 0.33, "CNN / BiLSTM+Attn / Transformer", "#e7f5dd"),
    ]
    eval_box = (0.5, 0.10, "10折分层交叉验证 (共享折划分, seed=42)\nTP/TN/FP/FN · Acc/P/R/F1/MCC · AUC · PR-AUC\nS7: 软投票 + Stacking(LR/XGBoost)", "#f3e8fd")

    for cx, cy, txt, col, *_ in boxes:
        ax.add_patch(FancyBboxPatch((cx - 0.16, cy - 0.07), 0.32, 0.14, boxstyle="round,pad=0.008",
                                    fc=col, ec="0.4", lw=1.2))
        ax.text(cx, cy, txt, ha="center", va="center", fontsize=9)
    for cx, cy, txt, col in paths:
        ax.add_patch(FancyBboxPatch((cx - 0.11, cy - 0.075), 0.22, 0.15, boxstyle="round,pad=0.008",
                                    fc=col, ec="0.4", lw=1.2))
        ax.text(cx, cy, txt, ha="center", va="center", fontsize=8.2)
    for cx, cy, txt, col in models:
        ax.add_patch(FancyBboxPatch((cx - 0.11, cy - 0.05), 0.22, 0.10, boxstyle="round,pad=0.008",
                                    fc=col, ec="0.4", lw=1.2))
        ax.text(cx, cy, txt, ha="center", va="center", fontsize=8.2)
    ax.add_patch(FancyBboxPatch((eval_box[0] - 0.2, eval_box[1] - 0.075), 0.4, 0.15,
                                boxstyle="round,pad=0.008", fc=eval_box[3], ec="0.4", lw=1.2))
    ax.text(eval_box[0], eval_box[1], eval_box[2], ha="center", va="center", fontsize=8.6)

    # 箭头
    for cx, _, _, _ in [(0.13, 0, 0, 0), (0.38, 0, 0, 0), (0.63, 0, 0, 0), (0.88, 0, 0, 0)]:
        ax.add_patch(FancyArrowPatch((0.5, 0.71), (cx, 0.64), arrowstyle="-|>", mutation_scale=12,
                                     color="0.45", lw=1.1))
        ax.add_patch(FancyArrowPatch((cx, 0.485), (cx, 0.39), arrowstyle="-|>", mutation_scale=12,
                                     color="0.45", lw=1.1))
        ax.add_patch(FancyArrowPatch((cx, 0.275), (0.5, 0.18), arrowstyle="-|>", mutation_scale=12,
                                     color="0.45", lw=1.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    out = os.path.join(C.FIG_DIR, "workflow.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print("实验流程图:", out)


def main():
    tab = load_pooled_table()
    tab.to_csv(os.path.join(C.RESULT_DIR, "comparison_all.csv"), index=False, encoding="utf-8-sig")
    print("\n===== 总对比表 (10折 mean±std / OOF混淆) =====")
    print(tab.to_string(index=False))
    plot_roc_pr()
    plot_bars(tab)
    plot_importance()
    plot_workflow()
    print("\n汇总完成: results/comparison_all.csv + figures/")


if __name__ == "__main__":
    main()
