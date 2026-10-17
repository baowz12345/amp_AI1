# -*- coding: utf-8 -*-
"""
12_scheme3_protbert.py   方案3: ProtBERT 微调(特征迁移版, CPU 友好)
==========================
SOTA 基线 / 迁移学习: 用蛋白质预训练语言模型 ProtBERT(Rostlab/prot_bert) 提取
序列表示([CLS], 1024 维), 再做线性探测(LR/SVM) 10 折 CV 分类。
注: 完整端到端微调 ProtBERT 需 GPU; 本脚本默认执行"冻结特征+线性探测"
(feature-based transfer), 该模式 CPU 可运行。若 --finetune 开启则进入
全参数微调(CPU 上极慢, 不建议)。
输出: results/protbert_rep.npy, results/metrics_S9_ProtBERT_*.csv, oof_*.npy
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def torch_no_grad(fn):
    def wrapper(*a, **k):
        import torch
        with torch.no_grad():
            return fn(*a, **k)
    return wrapper


def load_pipeline(device):
    from transformers import BertModel, BertTokenizer
    name = "Rostlab/prot_bert"
    print(f"加载 {name} (BertTokenizer slow + BertModel)...")
    tok = BertTokenizer.from_pretrained(name, do_lower_case=False)
    model = BertModel.from_pretrained(name)
    model.eval()
    model.to(device)
    return tok, model


@torch_no_grad
def extract_features(seqs, tok, model, device, batch=16):
    import torch
    reps = []
    for i in range(0, len(seqs), batch):
        chunk = seqs[i:i + batch]
        # ProtBERT 为氨基酸级 tokenizer: 序列需以空格分隔("G K V W ..."), 否则全为 UNK
        chunk = [" ".join(list(s)) for s in chunk]
        enc = tok(chunk, padding=True, truncation=True, max_length=100, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        cls = out.last_hidden_state[:, 0, :].cpu().numpy()   # [CLS] (B,1024)
        reps.append(cls)
    return np.concatenate(reps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=int, default=0, help=">0 时只用前 N 条做冒烟验证")
    ap.add_argument("--finetune", action="store_true", help="开启全参数微调(需GPU, CPU极慢)")
    args = ap.parse_args()

    C.setup_chinese_font()
    device = C.get_device()
    print(f"设备: {device}")
    torch_num_threads()

    df = C.load_dataset()
    if args.subset > 0:
        df = df.head(args.subset)
        print(f"[冒烟验证] 仅用前 {args.subset} 条")
    seqs, y = df["seq"].tolist(), df["label"].values

    if args.finetune:
        print("全参数微调模式: 需要 GPU, 本机 CPU 极慢, 建议仅 feature-based 模式。")
        run_finetune(seqs, y, device)
        return

    tok, model = load_pipeline(device)
    rep = extract_features(seqs, tok, model, device)
    np.save(os.path.join(C.RESULT_DIR, "protbert_rep.npy"), rep)
    print(f"ProtBERT 表示: {rep.shape}")

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    models = {
        "LR": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0)),
        "SVM": make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0, probability=True)),
    }
    res = C.run_sklearn_models_cv("S9_ProtBERT", rep, y, models, feature_names=None)
    for m, (dfo, _oof) in res.items():
        p = dfo[dfo["fold"] == "pooled_OOF"].iloc[0]
        print(f"  [{m}] AUC={p['AUC']:.4f} PR_AUC={p['PR_AUC']:.4f} MCC={p['MCC']:.4f} "
              f"Acc={p['Acc']:.4f} F1={p['F1']:.4f}")


def torch_num_threads():
    try:
        import torch
        torch.set_num_threads(max(1, os.cpu_count() // 2))
    except Exception:
        pass


def run_finetune(seqs, y, device):
    """全参数微调占位: CPU 上不实际运行, 提示用 GPU"""
    raise SystemExit("全参数微调需 GPU 环境(驱动/运行库兼容), 请使用 feature-based 模式或更换 GPU 机器。")


if __name__ == "__main__":
    main()
