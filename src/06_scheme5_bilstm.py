# -*- coding: utf-8 -*-
"""方案五: BiLSTM + 注意力池化 (10折CV)"""
import numpy as np

import common as C
import dl_common as D

if __name__ == "__main__":
    C.setup_chinese_font()
    df = C.load_dataset()
    seqs = df["seq"].tolist()
    y = df["label"].values
    X = C.seq_to_indices(seqs, D.MAX_LEN)
    print("== 方案五: BiLSTM + Attention (", X.shape, ") ==")
    D.run_deep_cv("BiLSTM", X, y, scheme_prefix="S5")
