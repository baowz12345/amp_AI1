# -*- coding: utf-8 -*-
"""
05_scheme4_cnn.py      = 方案四: 一维卷积神经网络 (CNN)
06_scheme5_bilstm.py   = 方案五: BiLSTM + 注意力池化
07_scheme6_transformer.py = 方案六: Transformer 自注意力
三者共用 dl_common.run_deep_cv, 输入均为索引编码 (CNN 在训练器内转 one-hot 由
dl_common.SeqDataset 处理, 本脚本直接传 one-hot 矩阵)。
"""
import os
import numpy as np
import pandas as pd

import common as C
import dl_common as D

if __name__ == "__main__":
    C.setup_chinese_font()
    df = C.load_dataset()
    seqs = df["seq"].tolist()
    y = df["label"].values
    # 本脚本即方案四: CNN (one-hot 输入)
    X = C.onehot_encoding(seqs, D.MAX_LEN)
    print("== 方案四: CNN (one-hot,", X.shape, ") ==")
    D.run_deep_cv("CNN", X, y, scheme_prefix="S4")
