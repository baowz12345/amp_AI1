# -*- coding: utf-8 -*-
"""
common.py
=========
公共模块: 中文字体、数据加载、特征工程(AAC/DPC/三肽/物理化学/One-hot/索引编码)、
评估指标(TP/TN/FP/FN/Acc/P/R/F1/MCC/AUC/PRAUC)、10折交叉验证框架、结果保存。
"""
import os
import sys
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
RESULT_DIR = os.path.join(BASE, "results")
FIG_DIR = os.path.join(RESULT_DIR, "figures")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# 标准氨基酸
AAS = "ACDEFGHIKLMNPQRSTVWY"
AA2IDX = {aa: i for i, aa in enumerate(AAS)}
PAD_IDX = len(AAS)
CV_SEED = 42
N_FOLDS = 10


def kmers3(seq, k=3):
    """3-mer 滑动窗口词袋(位置窗口滑过, 等价于字符 n-gram); 供 TF-IDF pipeline 与部署模型使用"""
    if len(seq) < k:
        return [seq]
    return [seq[i:i + k] for i in range(len(seq) - k + 1)]


def setup_chinese_font():
    """matplotlib 中文字体配置(Windows: 微软雅黑)"""
    from matplotlib import font_manager
    candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((c for c in candidates if c in available), "DejaVu Sans")
    plt.rcParams["font.sans-serif"] = [chosen] + [c for c in candidates if c != chosen]
    plt.rcParams["axes.unicode_minus"] = False
    return chosen


def load_dataset():
    """读取清洗后的数据集 CSV"""
    df = pd.read_csv(os.path.join(DATA_DIR, "dataset.csv"))
    return df


# ----------------------------------------------------------------------------
# 特征工程
# ----------------------------------------------------------------------------
def aac_features(seqs):
    """氨基酸组成 (20 维)"""
    X = np.zeros((len(seqs), len(AAS)), dtype=np.float32)
    for i, s in enumerate(seqs):
        c = Counter(s)
        n = len(s)
        for aa, cnt in c.items():
            X[i, AA2IDX[aa]] = cnt / n
    return X, [f"AAC_{aa}" for aa in AAS]


def dpc_features(seqs):
    """二肽组成 (400 维)"""
    X = np.zeros((len(seqs), len(AAS) ** 2), dtype=np.float32)
    for i, s in enumerate(seqs):
        n = len(s) - 1
        for j in range(n):
            d = s[j:j + 2]
            X[i, AA2IDX[d[0]] * len(AAS) + AA2IDX[d[1]]] += 1
        if n > 0:
            X[i] /= n
    names = [a + b for a in AAS for b in AAS]
    return X, [f"DPC_{n}" for n in names]


def tripeptide_counts(seqs):
    """三肽组成 (8000 维, 稀疏计数归一化)"""
    X = np.zeros((len(seqs), len(AAS) ** 3), dtype=np.float32)
    for i, s in enumerate(seqs):
        n = len(s) - 2
        if n <= 0:
            continue
        for j in range(n):
            t = s[j:j + 3]
            X[i, AA2IDX[t[0]] * len(AAS) ** 2 + AA2IDX[t[1]] * len(AAS) + AA2IDX[t[2]]] += 1
        X[i] /= n
    return X


def physico_features(seqs):
    """基于 peptides 库的物理化学特征 (15 维) + 组成特征可另行拼接"""
    from peptides import Peptide
    rows = []
    names = ["len", "mw", "charge_pH7", "pI", "hydrophobicity", "hydrophobic_moment",
             "aliphatic_index", "boman", "instability_index", "longest_run",
             "z1", "z2", "z3", "z4", "z5"]
    for s in seqs:
        p = Peptide(s)
        z = p.z_scales()
        try:
            hm = p.hydrophobic_moment()
        except Exception:
            hm = np.nan
        rows.append([len(s), p.molecular_weight(), p.charge(pH=7.0), p.isoelectric_point(),
                     p.hydrophobicity(), hm, p.aliphatic_index(), p.boman(),
                     p.instability_index(), p.longest_run(), *z])
    X = np.array(rows, dtype=np.float64)
    # 缺失值用列均值填充
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])
    return X, names


# ----------------------------------------------------------------------------
# 深度学习编码
# ----------------------------------------------------------------------------
def seq_to_indices(seqs, max_len):
    """嵌入模型输入: 序列 -> 索引 (pad=PAD_IDX), 形状 (N, max_len)"""
    X = np.full((len(seqs), max_len), PAD_IDX, dtype=np.int64)
    for i, s in enumerate(seqs):
        L = min(len(s), max_len)
        for j in range(L):
            X[i, j] = AA2IDX[s[j]]
    return X


def onehot_encoding(seqs, max_len):
    """CNN 输入: 序列 -> one-hot, 形状 (N, max_len, 20)"""
    X = np.zeros((len(seqs), max_len, len(AAS)), dtype=np.float32)
    for i, s in enumerate(seqs):
        L = min(len(s), max_len)
        for j in range(L):
            X[i, j, AA2IDX[s[j]]] = 1.0
    return X


# ----------------------------------------------------------------------------
# 评估指标
# ----------------------------------------------------------------------------
def compute_metrics(y_true, y_prob, threshold=0.5):
    from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                                 f1_score, matthews_corrcoef, precision_score,
                                 recall_score, roc_auc_score)
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "Acc": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "PR_AUC": average_precision_score(y_true, y_prob),
    }


def format_metric(per_fold_df):
    """per_fold_df: DataFrame, 每行是一折的指标 -> 返回 'mean±std' 字符串行"""
    out = {}
    for col in per_fold_df.columns:
        if col in ("fold", "model"):
            continue
        if pd.api.types.is_numeric_dtype(per_fold_df[col]):
            out[col] = f"{per_fold_df[col].mean():.4f}±{per_fold_df[col].std():.4f}"
    return out


def save_cv_results(name, fold_metrics_list, oof_prob, y_true, save_oof=True):
    """fold_metrics_list: 每折一个指标 dict; oof_prob: (N,) 正类概率; 保存 CSV 与 OOF"""
    df_folds = pd.DataFrame(fold_metrics_list)
    df_folds.insert(0, "fold", np.arange(1, len(fold_metrics_list) + 1))
    pooled = compute_metrics(y_true, oof_prob)
    df_pooled = pd.DataFrame([{**pooled, "fold": "pooled_OOF"}])
    df_out = pd.concat([df_folds, df_pooled], ignore_index=True)
    df_out.insert(0, "model", name)
    df_out.to_csv(os.path.join(RESULT_DIR, f"metrics_{name}.csv"), index=False, encoding="utf-8-sig")
    if save_oof:
        np.save(os.path.join(RESULT_DIR, f"oof_{name}.npy"), oof_prob)
    return df_out


def stratified_kfold(seed=CV_SEED, n_splits=N_FOLDS):
    from sklearn.model_selection import StratifiedKFold
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def run_sklearn_models_cv(name_prefix, X, y, models, feature_names=None, topk_importance=30):
    """统一跑 sklearn 模型的 10 折 CV。
    models: dict{短名: 估计器或Pipeline}
    返回: {短名: (fold_df, oof_prob)}; 同时保存 metrics CSV 与 OOF npy
    """
    from sklearn.model_selection import StratifiedKFold
    from scipy.sparse import issparse
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    is_text = isinstance(X, list) or (isinstance(X, np.ndarray) and X.dtype.kind in "OU")
    if is_text:
        X = np.asarray(X, dtype=object)
    else:
        X = X.astype(np.float64) if issparse(X) else np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    results = {}
    for mname, model in models.items():
        full = f"{name_prefix}_{mname}"
        fold_metrics, oof = [], np.zeros(len(y), dtype=np.float64)
        importances = []
        for tr_idx, te_idx in skf.split(X, y):
            m = model
            if hasattr(m, "clone"):
                from sklearn.base import clone
                m = clone(model)
            m.fit(X[tr_idx], y[tr_idx])
            prob = m.predict_proba(X[te_idx])[:, 1]
            oof[te_idx] = prob
            fold_metrics.append(compute_metrics(y[te_idx], prob))
            # 收集特征重要性
            est = m
            if hasattr(m, "named_steps"):
                est = m.named_steps.get("clf", m)
            if hasattr(est, "feature_importances_"):
                importances.append(est.feature_importances_)
        df_folds = pd.DataFrame(fold_metrics)
        df_folds.insert(0, "fold", np.arange(1, N_FOLDS + 1))
        pooled = compute_metrics(y, oof)
        df_out = pd.concat([df_folds, pd.DataFrame([{**pooled, "fold": "pooled_OOF"}])],
                           ignore_index=True)
        df_out.insert(0, "model", full)
        df_out.to_csv(os.path.join(RESULT_DIR, f"metrics_{full}.csv"), index=False, encoding="utf-8-sig")
        np.save(os.path.join(RESULT_DIR, f"oof_{full}.npy"), oof)
        results[mname] = (df_out, oof)
        if importances:
            imp_mean = np.mean(importances, axis=0)
            _save_importance(full, feature_names, imp_mean, topk_importance)
    return results


def _save_importance(model_full, feature_names, imp, topk):
    if feature_names is None:
        return
    df_imp = pd.DataFrame({"feature": feature_names, "importance": imp}).sort_values(
        "importance", ascending=False).head(topk).reset_index(drop=True)
    df_imp.to_csv(os.path.join(RESULT_DIR, f"importance_{model_full}.csv"), index=False, encoding="utf-8-sig")





# ----------------------------------------------------------------------------
# torch 设备探测
# ----------------------------------------------------------------------------
def get_device():
    import os as _os
    if _os.environ.get("AMP_FORCE_CPU", "0") == "1":
        import torch
        return torch.device("cpu")
    try:
        import torch
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    except Exception:
        return None


if __name__ == "__main__":
    setup_chinese_font()
    df = load_dataset()
    print("数据集:", df.shape, "正类:", int(df["label"].sum()))
    lens = df["len"].values
    print("P98 长度:", int(np.percentile(lens, 98)), "P95:", int(np.percentile(lens, 95)))
