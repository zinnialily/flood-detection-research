# Flood Detection from Street-Level Imagery

**Binary flood detection using hard negative mining (HNM) with progressive transfer learning.**
Target journal: *Computers & Geosciences* (Elsevier, ISSN 0098-3004).

> **Emphasis on high recall** — in emergency response, missing a real flood is far more costly than a false alarm.

---

## Overview

This project trains EfficientNetB0 and ResNet50 classifiers to detect flooding in street-level photos. The core contribution is a **hard negative mining pipeline** that identifies visually confusing non-flood images (e.g. swimming pools) and injects augmented copies back into training to reduce false positives.

The pipeline addresses three problems in the original CIBB 2026 baseline:
1. A double-rescaling preprocessing bug that rendered transfer learning non-functional
2. ResNet50 had no HNM counterpart, making the architectural comparison confounded
3. Evaluation lacked PR-AUC, bootstrap CIs, GradCAM, and severity-stratified recall

---

## Dataset

**Source:** Google Drive — `FloodingDataset2/`

```
FloodingDataset2/
  StreetFloodClasses/
    MajorFlood/      MinorFlood/     ModerateFlood/
    NoFlood/         parks_walkways/
  junk/
    Swimmingpool.zip, Cars.zip, Dogs.zip, ...  (12 distractor categories)
  processed_data/
    binary/
      train/  (flood/, non_flood/)   — 2,627 images (70%)
      val/    (flood/, non_flood/)   —   563 images (15%)
      test/   (flood/, non_flood/)   —   564 images (15%)
```

Splits are stratified by flood severity × swimming pool status (seed 42).

---

## Pipeline

| Step | Notebook | Script | Compute |
|------|----------|--------|---------|
| 1. Data exploration | `01_data_exploration.ipynb` | — | CPU |
| 2. Stratified splitting | `02_stratified_splitting.ipynb` | — | CPU |
| 3. Baseline EfficientNetB0 | `03_baseline_efficientnetb0.ipynb` | `scripts/train_baseline.py --arch efficientnet` | T4 GPU |
| 4. Baseline ResNet50 | `04_baseline_resnet50.ipynb` | `scripts/train_baseline.py --arch resnet50` | T4 GPU |
| 5a. Confounder analysis | `05a_confounder_analysis.ipynb` | `scripts/analyze_confounders.py` | T4 GPU |
| 5b. HNM — EfficientNetB0 | `05b_hnm_efficientnetb0.ipynb` | `scripts/train_hnm.py --arch efficientnet` | T4 GPU |
| 5c. HNM — ResNet50 | `05c_hnm_resnet50.ipynb` | `scripts/train_hnm.py --arch resnet50` | T4 GPU |
| 6. Evaluation | `06_evaluation.ipynb` | `scripts/evaluate.py` | CPU/GPU |

All GPU training runs on **Google Colab T4** via the VS Code Colab extension. CPU tasks can run locally.

### Execution order

```
03 → 04 → 05a → 05b → 05c → 06
```

After step 05a, the confounder FP table identifies which non-flood categories to mine. Copy the printed checkpoint path into `MODEL_PATH` in the next notebook before running.

---

## Scripts

```
scripts/
  utils.py               # seeding, model building, preprocessing, callbacks
  train_baseline.py      # two-phase progressive fine-tuning (Phase 1 + Phase 2)
  analyze_confounders.py # rank train/non_flood categories by FP rate
  train_hnm.py           # HNM with percentile or tau-sweep mining modes
  evaluate.py            # PR-AUC primary, bootstrap CI, McNemar, severity recall
  grad_cam.py            # GradCAM++ heatmaps for FN/FP/pool/correct sets
```

**Primary evaluation metric: PR-AUC** (not ROC-AUC). ROC-AUC can be misleadingly optimistic under class imbalance in disaster detection.

---

## Key Hyperparameters

| Parameter | EfficientNetB0 | ResNet50 |
|-----------|---------------|---------|
| Input size | 224 × 224 | 224 × 224 |
| Phase 1 LR | 1e-4 | 1e-4 |
| Phase 2 LR | 1e-5 | 1e-5 |
| HNM Phase 1 LR | 5e-5 | 5e-5 |
| Phase 1 trainable layers | last 30 | last 30 |
| Phase 2 frozen layers | first 50 | first 50 |
| HNM threshold (tau) | 0.3 | 0.3 |
| HNM augmentation | 5× | 5× |
| Batch size | 32 | 32 |
| Random seed | 42 | 42 |

---

## Preprocessing

**Critical:** Uses backbone-native preprocessing (`tf.keras.applications.efficientnet.preprocess_input` / `resnet50.preprocess_input`) with `include_preprocessing=False`. Do **not** use `ImageDataGenerator(rescale=1./255)` — this causes double-rescaling and breaks transfer learning.

A `verify_preprocessing()` assertion runs before every training loop and halts on misconfiguration.

---

## Setup

### Local (CPU tasks)

```bash
pip install -r requirements.txt
```

### Colab (GPU training)

```python
# At top of each training notebook
from google.colab import drive
drive.mount('/content/drive')

%pip install -q tf-keras-vis umap-learn

# Then run the script
!python scripts/train_baseline.py \
  --arch efficientnet \
  --data_dir /content/drive/MyDrive/FloodingDataset2 \
  --output_dir /content/drive/MyDrive/models
```

### Requirements

```
tensorflow>=2.16,<2.18
numpy>=1.26
pandas>=2.2
scikit-learn>=1.4
matplotlib>=3.8
seaborn>=0.13
Pillow>=10.0
scipy>=1.12
tf-keras-vis>=0.8
umap-learn>=0.5
```

---

## Repository Structure

```
imagevalidation2/
  notebooks/          Colab-ready .ipynb wrappers (one per pipeline step)
  scripts/            All training, mining, evaluation, and visualization logic
  results/
    figures/          PR curves, confusion matrices, GradCAM grids
    tables/           Metric CSVs, confounder FP rates, tau sweep results
    logs/             Training history CSVs
    predictions/      Per-image prediction CSVs
  methodology.mmd     Pipeline diagram (Mermaid)
  requirements.txt
```

---

## Data Leakage Controls

- HNM mining runs only on `train/non_flood/` — never on val or test
- Partition integrity assertion halts training if any overlap is detected
- Tau threshold selected on validation set; test set evaluated exactly once per model
- Calibration (if used) fitted on validation predictions, not test

---

## License

MIT

---

## Citation

> [Citation to be added upon acceptance]
