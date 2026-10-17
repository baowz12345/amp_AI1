# -*- coding: utf-8 -*-
"""
04_scheme3_word2vec.py
======================
方案三: 无监督词向量表示 (Word2Vec, 3-mer 词) + 机器学习分类器
  流程: 序列切分为 3-mer 词 -> 每折独立训练 Word2Vec(64维) -> 序列取词向量均值
        -> LR / SVM / RF 分类
  评估: 10折分层交叉验证(词向量训练仅在训练折内进行, 避免数据泄漏)
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import common as C

C.setup_chinese_font()
df = C.load_dataset()
seqs = df["seq"].tolist()
y = df["label"].values

W2V_DIM = 64


def make_corpus(seqs):
    return [[s[i:i + 3] for i in range(len(s) - 2)] for s in seqs]


def train_w2v(corpus):
    from gensim.models import Word2Vec
    return Word2Vec(corpus, vector_size=W2V_DIM, window=5, min_count=1, sg=1,
                    epochs=40, workers=4, seed=C.CV_SEED)


def encode(seqs, model):
    X = np.zeros((len(seqs), W2V_DIM), dtype=np.float32)
    for i, s in enumerate(seqs):
        words = [s[j:j + 3] for j in range(len(s) - 2)]
        vecs = [model.wv[w] for w in words if w in model.wv]
        if vecs:
            X[i] = np.mean(vecs, axis=0)
    return X


models = {
    "LR": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
    "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale",
                                               class_weight="balanced", probability=True,
                                               random_state=C.CV_SEED)),
    "RF": RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=3,
                                 class_weight="balanced", n_jobs=-1, random_state=C.CV_SEED),
}

print("== 10折交叉验证 (每折重训 Word2Vec) ==")
from sklearn.base import clone
skf = C.stratified_kfold()
all_corpus = make_corpus(seqs)
oof_all = {m: np.zeros(len(y)) for m in models}
fold_metrics_all = {m: [] for m in models}
for fold, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(len(y)), y), 1):
    w2v = train_w2v([all_corpus[i] for i in tr_idx])
    X = encode(seqs, w2v)
    for mname, model in models.items():
        m = clone(model)
        if not hasattr(m, "named_steps"):
            m = make_pipeline(StandardScaler(), m)
        m.fit(X[tr_idx], y[tr_idx])
        prob = m.predict_proba(X[te_idx])[:, 1]
        oof_all[mname][te_idx] = prob
        fold_metrics_all[mname].append(C.compute_metrics(y[te_idx], prob))
    if fold % 2 == 0:
        print(f"  第 {fold}/10 折完成")

results = {}
for mname in models:
    full = f"S3_{mname}"
    oof = oof_all[mname]
    df_folds = pd.DataFrame(fold_metrics_all[mname])
    df_folds.insert(0, "fold", np.arange(1, C.N_FOLDS + 1))
    pooled = C.compute_metrics(y, oof)
    df_out = pd.concat([df_folds, pd.DataFrame([{**pooled, "fold": "pooled_OOF"}])],
                       ignore_index=True)
    df_out.insert(0, "model", full)
    df_out.to_csv(os.path.join(C.RESULT_DIR, f"metrics_{full}.csv"), index=False, encoding="utf-8-sig")
    np.save(os.path.join(C.RESULT_DIR, f"oof_{full}.npy"), oof)
    results[mname] = (df_out, oof)
    line = {"方案": "S3_word2vec", "模型": mname}
    line.update(C.format_metric(df_folds))
    print(f"  [{mname}] AUC={line['AUC']}  F1={line['F1']}  MCC={line['MCC']}")

pd.DataFrame([{"方案": "S3_word2vec", "模型": m,
               **C.format_metric(v[0][v[0]["fold"] != "pooled_OOF"])}
              for m, v in results.items()]).to_csv(
    os.path.join(C.RESULT_DIR, "summary_S3.csv"), index=False, encoding="utf-8-sig")
print("方案三完成")
