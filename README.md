# Computational Intelligence Methods for Antimicrobial Peptide Classification: Ensemble Stacking and Multimodal Feature Fusion

This repository provides a reproducible benchmark of **nine computational-intelligence (CI) / machine-learning (ML) pipelines** for discriminating **animal-derived antimicrobial peptides (AMPs)** from **non-animal peptides**, curated from the Antimicrobial Peptide Database APD2024a release.

After standardization the curated dataset contains **3,208 sequences** (2,492 animal AMPs / 716 non-animal peptides). All pipelines are evaluated under a unified **10-fold stratified cross-validation** protocol with out-of-fold (OOF) predictions.

---

## 1. Repository layout

```
.
├── data/
│   ├── animalAMPs_APD2024a.fasta     # raw positive sequences from APD2024a
│   ├── naturalAMPs_APD2024a.fasta    # raw negative sequences from APD2024a
│   └── dataset.csv                  # cleaned 3,208-row benchmark table
├── src/
│   ├── common.py                    # shared IO, scoring, 10-fold CV helpers
│   ├── dl_common.py                 # shared PyTorch datasets, loops, EarlyStopping
│   ├── 01_explore_data.py           # dataset statistics
│   ├── 02_scheme1_physicochem.py    # Scheme 1: physicochemical + conventional ML
│   ├── 03_scheme2_kmer.py           # Scheme 2: k-mer + conventional ML
│   ├── 04_scheme3_word2vec.py      # Scheme 3: word2vec embeddings
│   ├── 05_scheme4_cnn.py            # Scheme 4: 1-D CNN
│   ├── 06_scheme5_bilstm.py         # Scheme 5: BiLSTM
│   ├── 07_scheme6_transformer.py    # Scheme 6: Transformer encoder
│   ├── 08_scheme7_ensemble.py       # Scheme 7: soft voting baseline
│   ├── 10_scheme6_tfidf.py         # Scheme 6 (TF-IDF): TF-IDF k-mer + LR/SVM
│   ├── 11_scheme2_cnn_bilstm.py    # Scheme 2: CNN-BiLSTM
│   ├── 12_scheme3_protbert.py       # Scheme 3: frozen ProtBERT + LR/SVM
│   ├── 13_scheme4_gnn.py            # Scheme 4: GCN / GAT on residue graph
│   ├── 14_scheme5_contrastive.py   # Scheme 5: contrastive self-supervised pretraining
│   ├── 15_scheme8_moe.py            # Scheme 8: mixture-of-experts with gating
│   ├── 16_scheme9_multimodal.py     # Scheme 9: sequence + Chou-Fasman secondary-structure
│   ├── 18_scheme10_pca.py           # Scheme 10: PCA / t-SNE / UMAP dimensionality reduction
│   ├── 19_feature_compare.py        # AAC / physicochem / DPC vs. combined comparison
│   ├── 22_feature_fusion.py         # early (concatenation) vs. late (probability) fusion
│   ├── 21_case_study.py             # held-out case study (LL-37, Magainin-2, Cecropin-A, Defensin-like)
│   ├── 17_summary9.py / 20_meanstd.py / 09_summary.py   # aggregation + 10-fold mean ± SD
│   └── 23_dataset_stats.py
├── results/
│   ├── meanstd_summary.csv          # 33-model 10-fold mean ± SD table (the main benchmark)
│   ├── comparison_schemes9.csv      # representative model per scheme
│   ├── feature_compare.csv          # single feature-type comparison
│   ├── feature_fusion.csv           # early vs late fusion
│   ├── case_study_pred.csv          # case-study predictions
│   ├── dataset_stats.csv
│   └── figures/                     # ROC/PR, PCA/t-SNE/UMAP, workflow diagrams
└── requirements.txt
```

---

## 2. Methods covered

| Scheme | Paradigm | Representative model |
|---|---|---|
| 1 | Hand-crafted descriptors + conventional ML | k-mer + XGBoost |
| 2 | End-to-end deep learning | CNN-BiLSTM |
| 3 | Frozen protein language model | ProtBERT + LR |
| 4 | Graph learning on residue graph | GCN |
| 5 | Contrastive self-supervised pretraining | BiGRU encoder + linear probe |
| 6 | Bag-of-words on peptides | TF-IDF k-mer + LR |
| 7 | Multi-model ensemble | Stacking (OOF) + LR meta-learner |
| 8 | Mixture-of-experts | Softmax-gated BiGRU + phys-chem |
| 9 | Sequence + secondary-structure multimodal | Chou-Fasman + BiGRU/Conv1D fusion |
| 10 | Dimensionality reduction | PCA (95% variance) → 322 PCs, t-SNE, UMAP |

---

## 3. Installation

```bash
conda create -n amp python=3.10 -y
conda activate amp
pip install -r requirements.txt
```

Key dependencies: `numpy`, `pandas`, `scikit-learn`, `scipy`, `xgboost`, `torch`, `transformers`, `biopython`, `networkx`, `torch_geometric`, `umap-learn`, `gensim`, `matplotlib`, `seaborn`.

> Deep-learning scripts (`05`–`07`, `11`–`16`) automatically fall back to CPU if no GPU is available. The ProtBERT script (`12`) downloads the HuggingFace model on first run.

## 4. Quick start

```bash
# 1) Convert the two raw APD2024a FASTA files into the cleaned dataset (data/dataset.csv)
#    (filtering rules: length 4-100, canonical amino acids, deduplication, class assignment)
python src/01_explore_data.py

# 2) Run individual schemes (each writes results/metrics_*.csv and OOF npy)
python src/02_scheme1_physicochem.py
python src/03_scheme2_kmer.py
python src/10_scheme6_tfidf.py
python src/12_scheme3_protbert.py
python src/14_scheme5_contrastive.py
python src/15_scheme8_moe.py
python src/18_scheme10_pca.py

# 3) Aggregate 10-fold mean ± SD
python src/20_meanstd.py

# 4) Feature-type comparison, fusion, case study
python src/19_feature_compare.py
python src/22_feature_fusion.py
python src/21_case_study.py
```

**Evaluation protocol (anti-leakage).** All models are evaluated by **10-fold stratified cross-validation** (shuffle, seed = 42); a single random train/test split is not used. Every data-dependent transformation — variance filter and mutual-information feature selection (Scheme 2), the TF-IDF vocabulary (Scheme 6), feature standardization, and principal-component analysis (Schemes 9-10 and the feature-type benchmark) — is fitted **inside each training fold only** and then applied to the held-out fold, so that no test information enters the training procedure.

## 4b. Inference with the trained models

Pre-trained model weights are provided in `models/` (joblib format) and are applied to **any new FASTA/CSV file** with a single command:

```bash
# 3-mer TF-IDF + logistic regression (fast, interpretable; recommended default)
python predict.py --input my_peptides.fasta --model S12_TFIDF_LR --out predictions.csv

# k-mer composition + feature selection + XGBoost
python predict.py --input my_peptides.fasta --model S2_XGB --out predictions.csv
```

The output CSV contains, for each sequence: `id`, `seq`, `prob_AMP` (probability of being an animal AMP) and `prediction` (AMP / non-AMP). The weights are regenerated with `python export_models.py` (full-data training, for deployment only; the paper reports 10-fold CV performance).

## 5. Headline results (10-fold CV mean ± SD)

| Model | AUC | PR-AUC | MCC |
|---|---|---|---|
| Stacking (LR meta-learner) | **0.965 ± 0.004** | **0.990 ± 0.002** | **0.755 ± 0.015** |
| k-mer + XGBoost (best single) | 0.955 ± 0.015 | 0.985 ± 0.006 | 0.755 ± 0.043 |
| CNN-BiLSTM | 0.932 ± 0.018 | 0.975 ± 0.011 | 0.675 ± 0.067 |
| ProtBERT + LR | 0.930 ± 0.021 | 0.975 ± 0.010 | 0.686 ± 0.053 |
| GCN | 0.898 ± 0.027 | 0.963 ± 0.013 | 0.590 ± 0.038 |
| Contrastive + LR | 0.895 ± 0.025 | 0.961 ± 0.011 | 0.581 ± 0.069 |

Late (decision-level) fusion (AUC = 0.958 ± 0.013) marginally outperforms early (feature-level) concatenation (0.957 ± 0.014) on XGBoost.

## 6. Citation

If you use this benchmark, please cite the accompanying manuscript.
