# -*- coding: utf-8 -*-
"""20_meanstd.py  计算全部模型 10 折均值±std 汇总, 供 IEEE 版表格/正文使用
输出: results/meanstd_summary.csv (model, 每指标 mean/std)
"""
import os
import sys
import glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

COLS = ["TP", "FP", "TN", "FN", "Acc", "Precision", "Recall", "F1", "MCC", "AUC", "PR_AUC"]


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(C.RESULT_DIR, "metrics_S*.csv"))):
        name = f[len(C.RESULT_DIR) + 9:-len(".csv")]
        df = pd.read_csv(f)
        fld = df[df["fold"] != "pooled_OOF"]
        pooled = df[df["fold"] == "pooled_OOF"].iloc[0]
        if fld.empty:
            continue
        rec = {"model": name}
        for c in COLS:
            vals = fld[c].astype(float)
            rec[c + "_mean"] = vals.mean()
            rec[c + "_std"] = vals.std()
            rec[c + "_pooled"] = pooled[c]
        rows.append(rec)
    out = pd.DataFrame(rows).sort_values("AUC_mean", ascending=False)
    out.to_csv(os.path.join(C.RESULT_DIR, "meanstd_summary.csv"), index=False, encoding="utf-8-sig")
    print(out[["model", "AUC_mean", "AUC_std", "PR_AUC_mean", "MCC_mean",
               "Acc_mean", "F1_mean", "AUC_pooled"]].to_string(index=False))


if __name__ == "__main__":
    main()
