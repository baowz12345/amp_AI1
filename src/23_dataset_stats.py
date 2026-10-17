# -*- coding: utf-8 -*-
"""Dataset statistical description for the IEEE report.

Computes length-distribution and amino-acid-composition statistics of the
cleaned APD2024a benchmark dataset and generates two English-labeled figures:
  fig12_len_dist.png  -- sequence-length distribution (pos vs neg)
  fig13_aa_comp.png   -- amino-acid composition (pos vs neg)
Outputs dataset_stats.csv (per-class length statistics + AA composition).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data", "dataset.csv")
RES = os.path.join(BASE, "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)

AAS = list("ACDEFGHIKLMNPQRSTVWY")

df = pd.read_csv(DATA, encoding="utf-8-sig")
pos = df[df["label"] == 1]
neg = df[df["label"] == 0]

# ---- per-class length statistics ----
rows = []
for name, sub in [("positive", pos), ("negative", neg)]:
    L = sub["len"].values
    rows.append({
        "class": name, "n": len(sub),
        "len_mean": L.mean(), "len_std": L.std(ddof=1),
        "len_min": L.min(), "len_q25": np.percentile(L, 25),
        "len_median": np.median(L), "len_q75": np.percentile(L, 75),
        "len_p95": np.percentile(L, 95), "len_max": L.max(),
    })
stats = pd.DataFrame(rows)

# ---- per-class amino-acid composition (relative frequency) ----
def aac_freq(seqs):
    cnt = {a: 0 for a in AAS}
    total = 0
    for s in seqs:
        for a in s:
            if a in cnt:
                cnt[a] += 1
                total += 1
    return {a: (cnt[a] / total if total else 0.0) for a in AAS}

freq_pos = aac_freq(pos["seq"].tolist())
freq_neg = aac_freq(neg["seq"].tolist())
comp = pd.DataFrame({"aa": AAS, "positive": [freq_pos[a] for a in AAS],
                     "negative": [freq_neg[a] for a in AAS]})
comp["diff"] = comp["positive"] - comp["negative"]
comp = comp.sort_values("positive", ascending=False).reset_index(drop=True)
comp.to_csv(os.path.join(RES, "dataset_stats.csv"),
            index=False, encoding="utf-8-sig")

# ---- Fig. 12: length distribution ----
fig, ax = plt.subplots(figsize=(7.6, 3.4), dpi=200)
bins = np.arange(0, 110, 5)
ax.hist(pos["len"], bins=bins, alpha=0.62, color="#4C72B0",
        label=f"Positive (n={len(pos)})")
ax.hist(neg["len"], bins=bins, alpha=0.62, color="#DD8452",
        label=f"Negative (n={len(neg)})")
ax.axvline(pos["len"].median(), color="#2C3E50", ls="--", lw=1.1)
ax.axvline(neg["len"].median(), color="#2C3E50", ls=":", lw=1.1)
ax.set_xlabel("Sequence length (residues)")
ax.set_ylabel("Number of sequences")
ax.set_title("Sequence-length distribution (cleaned APD2024a dataset)")
ax.legend(fontsize=8.5)
ax.set_xlim(0, 105)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig12_len_dist_en.png"))
plt.close(fig)

# ---- Fig. 13: amino-acid composition ----
fig, ax = plt.subplots(figsize=(7.6, 3.6), dpi=200)
x = np.arange(len(comp))
w = 0.38
ax.bar(x - w / 2, comp["positive"], w, color="#4C72B0", label="Positive")
ax.bar(x + w / 2, comp["negative"], w, color="#DD8452", label="Negative")
ax.set_xticks(x)
ax.set_xticklabels(comp["aa"], fontsize=9)
ax.set_ylabel("Relative frequency")
ax.set_xlabel("Amino acid (sorted by positive-class frequency)")
ax.set_title("Amino-acid composition of animal vs non-animal peptides")
ax.legend(fontsize=8.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig13_aa_comp_en.png"))
plt.close(fig)

print(stats.to_string(index=False))
print()
print(comp.head(10).to_string(index=False))
