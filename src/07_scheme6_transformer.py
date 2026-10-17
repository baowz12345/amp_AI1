# -*- coding: utf-8 -*-
"""方案六: Transformer 自注意力 (10折CV)"""
import numpy as np

import common as C
import dl_common as D

if __name__ == "__main__":
    C.setup_chinese_font()
    df = C.load_dataset()
    seqs = df["seq"].tolist()
    y = df["label"].values
    X = C.seq_to_indices(seqs, D.MAX_LEN)
    print("== 方案六: Transformer (", X.shape, ") ==")
    D.run_deep_cv("Transformer", X, y, scheme_prefix="S6")
