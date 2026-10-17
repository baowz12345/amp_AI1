# -*- coding: utf-8 -*-
"""
01_explore_data.py
==================
解析 APD2024a 的两个抗菌肽(AMP) FASTA 数据集, 完成:
  1) 统计序列数量、长度分布
  2) 检查两数据集的重叠关系(animal 是否为 natural 的子集)
  3) 清洗序列(标准氨基酸字母表、长度过滤、去重)
  4) 构建二分类数据集: 阳性=动物源AMP(animal), 阴性=非动物源AMP(natural\animal)
  5) 输出清洗后的数据集 CSV 与数据概况图
"""
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as C

C.setup_chinese_font()

BASE = C.BASE
DATA_DIR = C.DATA_DIR
RESULT_DIR = C.RESULT_DIR
FIG_DIR = C.FIG_DIR

STD_AA = set("ACDEFGHIKLMNPQRSTVWY")
MIN_LEN, MAX_LEN = 4, 100


def parse_fasta(path):
    """解析 FASTA, 跳过 '>Your search led to N peptides' 之类的统计头"""
    seqs, ids = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        cur_id, cur_seq = None, []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None and cur_seq:
                    seqs.append("".join(cur_seq).upper())
                    ids.append(cur_id)
                cur_id = line[1:].strip()
                # 跳过统计行(不含 AP 编号)
                if not re.match(r"^AP\d+", cur_id):
                    cur_id = None
                cur_seq = []
            else:
                if cur_id is not None:
                    cur_seq.append(line)
        if cur_id is not None and cur_seq:
            seqs.append("".join(cur_seq).upper())
            ids.append(cur_id)
    return ids, seqs


def clean(seq):
    """只保留标准氨基酸字母; 清洗后长度落在 [MIN_LEN, MAX_LEN] 才保留"""
    s = "".join(ch for ch in seq if ch in STD_AA)
    if MIN_LEN <= len(s) <= MAX_LEN:
        return s
    return None


def main():
    animal_ids, animal_seqs = parse_fasta(os.path.join(DATA_DIR, "animalAMPs_APD2024a.fasta"))
    natural_ids, natural_seqs = parse_fasta(os.path.join(DATA_DIR, "naturalAMPs_APD2024a.fasta"))
    print(f"animalAMPs 原始: {len(animal_seqs)} 条")
    print(f"naturalAMPs 原始: {len(natural_seqs)} 条")

    # 重叠关系
    s_animal = set(animal_seqs)
    s_natural = set(natural_seqs)
    print(f"animal ⊆ natural: {s_animal <= s_natural}")
    print(f"animal 中不在 natural 的序列数: {len(s_animal - s_natural)}")
    print(f"natural 中不在 animal 的序列数: {len(s_natural - s_animal)}")

    # 清洗
    animal_clean = [clean(s) for s in s_animal]
    animal_clean = [s for s in animal_clean if s]
    nonanimal_raw = list(s_natural - s_animal)
    nonanimal_clean = [clean(s) for s in nonanimal_raw]
    nonanimal_clean = [s for s in nonanimal_clean if s]
    print(f"清洗后 阳性(动物源): {len(animal_clean)}  阴性(非动物源): {len(nonanimal_clean)}")

    # 构建数据集
    df = pd.DataFrame({
        "seq": animal_clean + nonanimal_clean,
        "label": [1] * len(animal_clean) + [0] * len(nonanimal_clean),
        "source": ["animal"] * len(animal_clean) + ["non_animal"] * len(nonanimal_clean),
    })
    df["len"] = df["seq"].apply(len)
    df = df.drop_duplicates(subset="seq").reset_index(drop=True)
    print(f"最终数据集: {len(df)} 条, 正类={df['label'].sum()}, 负类={(1-df['label']).sum()}")
    print("长度统计:\n", df["len"].describe().round(2).to_string())

    df.to_csv(os.path.join(DATA_DIR, "dataset.csv"), index=False, encoding="utf-8-sig")

    # 数据概况图: 长度分布(按类别) + 类别占比
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for lab, col, name in [(1, "#d62728", "动物源 AMP"), (0, "#1f77b4", "非动物源 AMP")]:
        lens = df.loc[df["label"] == lab, "len"]
        axes[0].hist(lens, bins=40, alpha=0.65, label=f"{name}(n={len(lens)})", color=col)
    axes[0].set_xlabel("序列长度 (aa)")
    axes[0].set_ylabel("频数")
    axes[0].set_title("序列长度分布")
    axes[0].legend()
    axes[1].bar(["动物源 AMP", "非动物源 AMP"], [df['label'].sum(), (1 - df['label']).sum()],
                color=["#d62728", "#1f77b4"])
    axes[1].set_ylabel("序列数")
    axes[1].set_title("类别样本量")
    for i, v in enumerate([df['label'].sum(), (1 - df['label']).sum()]):
        axes[1].text(i, v + 15, str(v), ha="center", fontsize=11)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "data_overview.png")
    plt.savefig(out, dpi=150)
    print("概况图已保存:", out)
    print("清洗后数据已保存:", os.path.join(DATA_DIR, "dataset.csv"))


if __name__ == "__main__":
    main()
