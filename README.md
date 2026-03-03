# PolaFuse
# PolaFusion 🔥
### *SemEval-2026 Task 9 — Multilingual Polarization Detection*

<p align="center">
  <img src="https://img.shields.io/badge/SemEval-2026-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Task-Multilingual%20Polarization-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Languages-22-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Platform-Kaggle%20A100-red?style=for-the-badge" />
</p>

<p align="center">
  <b>Subtask 1 — Macro-F1: 0.800 &nbsp;|&nbsp; Subtask 2 — Macro-F1: 0.576 &nbsp;|&nbsp; Subtask 3 — Macro-F1: 0.502</b>
</p>

---

## 📌 Overview

**PolaFusion** is our system for [SemEval-2026 Task 9](https://semeval.github.io/SemEval2026/tasks), which requires detecting and characterizing polarization in social media posts across **22 languages**. The task has three hierarchical subtasks:

| Subtask | Description | Type | Labels |
|---------|-------------|------|--------|
| **ST1** | Is the post polarized? | Binary classification | `polarized / non-polarized` |
| **ST2** | What type of polarization? | Multi-label classification | `Political, Racial/Ethnic, Religious, Gender/Sexual, Other` |
| **ST3** | What rhetorical manifestation? | Multi-label classification | `Stereotype, Vilification, Dehumanization, Extreme Language, Lack of Empathy, Invalidation` |

All subtasks are evaluated with **Macro-F1**, which treats every class equally regardless of frequency — making pervasive class imbalance the central challenge.

---

## 🏗️ System Architecture

PolaFusion is built around a **hierarchical gating** design. A binary Gatekeeper model (ST1) first determines whether a post is polarized. Only polarized posts are passed downstream to the Type Specialist (ST2) and Manifestation Specialist (ST3), both of which are trained exclusively on polarized instances.

```
Input Post
    │
    ▼
┌─────────────────────────────┐
│   GATEKEEPER  (Subtask 1)   │  ← 8-model ensemble
│   mDeBERTa ×5 + XLM-R ×3   │     soft-vote aggregation
└─────────────┬───────────────┘
              │
      ┌───────┴────────┐
      │                │
  ŷ₁ = 0           ŷ₁ = 1
  (non-polar)      (polarized)
      │                │
  All-zero         ┌───┴──────────────────────┐
  ST2 / ST3        │   TYPE SPECIALIST (ST2)  │  ← 8-model ensemble
  output           │   5-label multi-label    │
                   └───┬──────────────────────┘
                       │
               ┌───────┴──────────────────────────┐
               │  MANIFESTATION SPECIALIST (ST3)  │  ← 8-model ensemble
               │  6-label multi-label             │
               └──────────────────────────────────┘
```

### 🔢 The 8-Model Mega-Ensemble

Each specialist is a soft-vote ensemble of **8 independently trained models**:

| Model | Architecture | Folds | Strength |
|-------|-------------|-------|----------|
| `microsoft/mdeberta-v3-base` | Disentangled attention | 5-fold CV | Morphologically complex languages |
| `xlm-roberta-large` | Large multilingual encoder | 3-fold CV | Low-resource languages, broader coverage |

Final predictions are made by averaging raw sigmoid probabilities across all 8 models before thresholding — no learned weighting required.

---

## ⚖️ Handling Class Imbalance

Severe class imbalance is the defining challenge of this dataset. Some labels appear in under 2% of training instances (e.g., *Dehumanization* in Odia, *Lack of Empathy* in Hausa). We address this with three complementary strategies:

### 1. Hierarchical Gating
Training ST2/ST3 specialists only on polarized data eliminates the majority of all-zero label vectors that would otherwise dominate multi-label training.

### 2. Macro-F1-Aware Augmentation (Qwen3-235B)
Synthetic data is generated using **Qwen3-235B-A22B** via the NVIDIA NIM API, but **only** when two conditions hold simultaneously:

```
augment = True  iff:
    (minority/majority ratio < threshold)  AND
    (per-label dev-set Macro-F1 < threshold)
```

The number of synthetic examples per triggered pair is capped at:
```
s = min(0.8 × n_max, 2 × n_c) − n_c
```
This prevents augmented data from dominating the label distribution and avoids more than doubling any single class.

### 3. Focal Loss
Both mDeBERTa and XLM-R are trained with **Focal Loss** (γ=2), which down-weights easy negatives and forces the model to focus on hard minority-class examples.

---

## 📊 Results

### Official Scores

| Subtask | Macro-F1 | vs. Single-Model Baseline |
|---------|----------|--------------------------|
| ST1 — Polarization Detection | **0.800** | +0.450 |
| ST2 — Type Classification | **0.576** | +0.131 |
| ST3 — Manifestation ID | **0.502** | +0.351 |

### Per-Language Highlights (ST1)

| Best | F1 | Hardest | F1 |
|------|----|---------|----|
| Nepali | 0.909 | Italian | 0.671 |
| Chinese | 0.902 | German | 0.719 |
| Burmese | 0.874 | Khmer | 0.703 |

> Italian scores low due to polarization expressed through **irony and understatement** — patterns that don't transfer well from cross-lingual pretraining on more direct expressions.
> Hausa scores low due to **shallow representation in both mDeBERTa and XLM-R pretraining corpora**.

---

## 🗂️ Repository Structure

```
polafusion/
│
├── README.md                          ← You are here
│
└── src/
    └── notebooks/
        ├── augmentation.ipynb         ← Step 1: Build augmentation plan + generate synthetic data
        ├── training.ipynb             ← Step 2: Train all 6 model variants (mDeBERTa + XLM-R × 3 subtasks)
        └── inference.ipynb            ← Step 3: Load all 8 models, soft-vote, generate submissions
```

---

## 📓 Notebook Guide

### `augmentation.ipynb`

> **Purpose:** Identify which language-label pairs need synthetic data, then generate it using Qwen3-235B.

| Cell | What it does |
|------|-------------|
| 1 | Builds the ST1 augmentation plan — scans training data and dev-set F1 scores to decide which languages need augmentation |
| 2 | Builds the ST3 augmentation plan (manifestation labels) |
| 3 | Builds the ST2 augmentation plan (polarization type labels) |
| 4 | Calls Qwen3-235B for ST1 augmentation — generates paraphrases preserving label semantics |
| 5 | Calls Qwen3-235B for ST2/ST3 augmentation with manifestation-aware prompts |
| 6 | Calls Qwen3-235B for ST2 type augmentation |

**⚠️ Important:** Replace the `api_key` in augmentation cells with your own [NVIDIA NIM API key](https://integrate.api.nvidia.com). The key in the notebook has been rotated.

**Output files:** `subtask1_augmented.csv`, `subtask2_augmented.csv`, `subtask3_augmented.csv`

---

### `training.ipynb`

> **Purpose:** Fine-tune mDeBERTa-v3-base (5-fold) and XLM-RoBERTa-large (3-fold) for all three subtasks.

| Cell | Model | Subtask | Folds | Key Details |
|------|-------|---------|-------|-------------|
| 1 | mDeBERTa-v3-base | ST1 | 5 | Binary classification, `KFold`, Focal Loss |
| 2 | mDeBERTa-v3-base | ST2 | 5 | Multi-label (5), `StratifiedKFold` by language+sum |
| 3 | mDeBERTa-v3-base | ST3 | 5 | Multi-label (6), `StratifiedKFold` by language+sum |
| 4 | XLM-RoBERTa-large | ST1 | 3 | Binary, gradient accumulation=2, batch=8 |
| 5 | XLM-RoBERTa-large | ST2 | 3 | Multi-label (5), `save_only_model=True` for disk space |
| 6 | XLM-RoBERTa-large | ST3 | 3 | Multi-label (6), same disk-safe settings |

**Training configuration:**

```python
# mDeBERTa settings
LEARNING_RATE = 2e-5
BATCH_SIZE    = 16
EPOCHS        = 3–4
MAX_LEN       = 128

# XLM-R-large settings (memory-constrained)
LEARNING_RATE          = 1e-5
BATCH_SIZE             = 8
GRADIENT_ACCUMULATION  = 2   # effective batch = 16
EPOCHS                 = 3–4
```

Each fold's heavy checkpoints are deleted immediately after saving the clean best model to manage Kaggle disk quotas.

---

### `inference.ipynb`

> **Purpose:** Load all 8 trained models per subtask, run soft-vote ensemble inference, apply thresholds, and write submission files.

| Step | Description |
|------|-------------|
| Config | Set paths to all 8 model directories per subtask |
| `run_model()` | Load one model, return sigmoid probabilities |
| `soft_vote()` | Average probabilities across all 8 models |
| `forced_argmax()` | For polarized posts with all-zero predictions, assign the highest-prob label |
| ST1 inference | Predict polarization for all test posts |
| ST2 inference | Predict type labels only for ST1-positive posts |
| ST3 inference | Predict manifestation labels only for ST1-positive posts |

**Binarization thresholds:**

```python
THRESH_ST1 = 0.50   # balanced threshold
THRESH_ST2 = 0.35   # lower to improve recall on rare types
THRESH_ST3 = 0.30   # lowest — manifestations are most severely imbalanced
```

---

## ⚙️ Setup & Requirements

### Environment
All training was done on **Kaggle's NVIDIA A100 (40GB)** GPU tier. The notebooks are written for the Kaggle environment but can be adapted to any GPU machine with path adjustments.

### Dependencies

```bash
pip install transformers==4.40.0
pip install torch>=2.0.0
pip install scikit-learn pandas numpy tqdm
pip install openai   # for augmentation (NVIDIA NIM client)
```

### Data
The dataset is the **POLAR benchmark** ([Naseem et al., 2026](https://arxiv.org/abs/2505.20624)), available through the [SemEval-2026 Task 9 organizers](https://semeval.github.io/SemEval2026/tasks). Update the `DATA_PATH` variables in each notebook to point to your local copy.

---

## 🧪 Ablation Summary

| Configuration | ST1 F1 | ST2 F1 | ST3 F1 |
|--------------|--------|--------|--------|
| Single mDeBERTa (1 fold) | 0.350 | 0.445 | 0.151 |
| mDeBERTa 5-fold + XLM-R 1-fold | 0.740 | 0.567 | 0.433 |
| + Macro-F1-aware augmentation | 0.770 | 0.572 | 0.465 |
| + Per-label threshold tuning | — | 0.557 ↓ | 0.457 ↓ |
| + Forced argmax correction | — | — | 0.480 |
| **Final (all data + augmentation)** | **0.800** | **0.576** | **0.502** |

> **Key finding:** Ensembling is the dominant gain (+0.390 on ST1). Threshold tuning consistently **hurts** generalization — the per-language-label validation sets are too small for reliable threshold estimation.

---

## 📎 Citation

If you use this code or build on our work, please cite:

```bibtex
@inproceedings{polafusion2026,
  title     = {PolaFusion at SemEval-2026 Task 9: Ensemble Transformers with
               Targeted Augmentation for Multilingual Polarization Detection},
  author    = {Anonymous},
  booktitle = {Proceedings of the 20th International Workshop on Semantic Evaluation
               (SemEval-2026)},
  year      = {2026},
  publisher = {Association for Computational Linguistics}
}
```

---

## 📄 License

This repository is released for research purposes. The POLAR dataset is subject to the licensing terms of the SemEval-2026 Task 9 organizers.

---

<p align="center">Made with ❤️ for SemEval-2026 &nbsp;|&nbsp; Trained on Kaggle A100 &nbsp;|&nbsp; Powered by 🤗 Transformers</p>
