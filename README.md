# First-Pass Flood Detection from Street-Level Imagery: Hard Negative Mining for Confounder-Robust Screening

Binary flood classification using progressive transfer learning and explicit hard negative mining (HNM).
Target journal: *Computers & Geosciences* (Elsevier, ISSN 0098-3004).

> **High recall is the primary objective.** In emergency response, a missed flood is far more costly than a false alarm.

---

## Relevant Prior Work

**AlleyFloodNet (Lee et al., 2025)** applies ConvNeXt-Large to street-level flood binary classification in urban alleys, achieving 96.56% accuracy, 97.67% recall, and an F1 of 96.55% — establishing that high-accuracy binary detection is achievable on curated street-level imagery.
*Gap:* Does not study false positive rates on specific confounding categories (swimming pools, wet roads) and does not apply hard negative mining or report PR-AUC.

**CrisisMMD (Alam et al., 2018)** is an 18,082-image multimodal Twitter dataset from seven 2017 natural disasters annotated for informativeness, humanitarian category, and damage severity.
*Gap:* Not flood-specific; social media domain introduces large intra-class variance and unlabelled near-duplicates. Binary informativeness framing does not target the precision-recall tradeoff relevant to operational flood screening.

**MediaEval 2020 Flood Task (Papadimos et al., 2023)** is a binary relevance classification challenge on 7,698 Italian tweets from 2017–2019 flood events; the best multimodal GNN approach achieved F1 = 0.5379.
*Gap:* Open-domain social media with 21% positive prevalence makes the task fundamentally different from street-level camera screening. Low absolute F1 reflects classifier confusion from non-flood Twitter content, not a property of flood imagery itself.

**Khan et al. (2023)** propose a modified ResNet50 for UAV flood classification achieving 96.43% accuracy and demonstrate deployment on NVIDIA Jetson Nano (820 ms inference at 6.9 W).
*Gap:* UAV/aerial domain; does not address street-level confounders. No PR-AUC reported, no hard negative mining, and single-seed results on a small test set without confidence intervals.

**OHEM — Shrivastava et al. (2016)** introduce Online Hard Example Mining for object detection, showing that focusing gradient updates on misclassified examples outperforms random sampling and reduces training cost.
*Gap:* Designed for two-stage detectors on PASCAL VOC; not demonstrated for binary flood classification or for the swimming-pool / wet-road confounders that are specific to street-level imagery.

**Focal Loss — Lin et al. (2017)** propose a soft down-weighting of well-classified examples via a modulating factor (1 − p_t)^γ, underpinning RetinaNet.
*Gap:* Focal loss modulates gradients within each epoch's existing data distribution. Unlike explicit HNM, it does not permanently add augmented confounders to the training set. The two approaches are complementary; their relative contribution in flood detection has not been evaluated.

**FloodTrace (Dyken et al., 2024)** is a crowdsourced annotation web application for flood extents with a median annotation time less than half the state of the art.
*Gap:* Addresses labelling bottleneck but not classifier robustness to confounders.

**Esparza et al. (2022)** analyse sample, spatial, and demographic biases in crowdsourced flood reports, showing systematic under-representation of rural and economically marginalised communities.
*Gap:* Identifies bias in crowdsourced training data but does not propose a mitigation at the classifier level.

---

## Motivation

### The visual confounder problem

A largely underexplored failure mode in flood binary classification is the **false positive from visually confusing non-flood categories**. Swimming pools share three properties with flooded streets: an open reflective water surface at approximately ground level, proximity to urban infrastructure, and similar colour statistics under overcast conditions. Wet roads and reflective pavement produce specular reflections that can mimic standing water, particularly at night (Schumann et al., 2023). Rain-obscured images introduce surface water appearances on streets, windows, and lenses. None of these confounders are systematically represented in existing flood detection datasets, and none of the reviewed papers explicitly report false positive rates on confounding categories separately from aggregate precision.

Misclassifying a swimming pool as flooded would trigger a false emergency dispatch. This category of error — high-confidence false positive on a semantically distinct but visually similar class — is precisely what hard negative mining is designed to correct.

### The confirmation bias problem in self-mining

A known limitation of offline HNM is **confirmation bias**: a model that mines its own hard negatives identifies examples that are hard specifically for its current representational state, potentially reinforcing its existing biases rather than resolving genuine semantic confusion (Shrivastava et al., 2016). Cross-seed mining — using one model checkpoint to mine negatives for a differently-initialised model — is a proposed mitigation that has not been evaluated in disaster detection.

### Statistically underpowered comparisons

Most flood detection papers report single-run results on test sets of 500–600 images without confidence intervals. On 564 images, a 5-point accuracy difference corresponds to approximately 28 images — a quantity sensitive to random split composition. McNemar's test and bootstrap CIs are necessary to distinguish genuine improvements from sampling variation. This project applies both.

---

## Key Contributions

1. **Explicit offline HNM for street-level flood classification.** No prior paper applies mining-and-augmentation of specific confounder categories (swimming pools, wet roads) to the flood binary classification problem. This project implements offline HNM with 5× augmentation injection for both EfficientNetB0 and ResNet50, enabling a direct architectural comparison of HNM benefit.

2. **Two-phase progressive fine-tuning protocol.** Progressive unfreezing is known to mitigate catastrophic forgetting (Lyu et al., 2025; Neupane et al., 2025) but has not been evaluated on flood datasets. This project ablates phase boundaries for both backbones.

3. **Systematic confounder FP analysis.** False positive rates on each non-flood category (swimming pool, wet road, animals, vehicles, buildings) are reported separately with Clopper-Pearson binomial CIs. The swimming pool result of 0/15 carries a 95% CI of approximately [0%, 21.8%] — reported honestly, not as "complete elimination."

4. **Extended-training-without-HNM ablation.** The HNM retraining effectively doubles the training budget. An explicit control (same total epochs, no injection) isolates the HNM contribution from the benefit of additional training time. This is the single most important ablation in the pipeline.

5. **PR-AUC as primary metric.** ROC-AUC is known to be misleadingly optimistic under class imbalance (Davis & Goadrich, 2006). PR-AUC directly captures the precision-recall tradeoff that matters for screening: high recall subject to a manageable false positive rate.

6. **Statistically rigorous evaluation.** Multi-seed runs (seeds 42, 123, 256, 512, 1024) with bootstrap CIs and McNemar's test for pairwise model comparisons. Fisher's exact test for swimming pool FP rate comparisons.

7. **Severity-stratified recall.** The dataset preserves flood severity labels (MajorFlood, ModerateFlood, MinorFlood). Recall broken down by severity identifies which flood presentations the model misses most, which has direct operational implications for emergency response triage.

---

## Methods

### Architecture

```
Input (224 × 224 × 3)
  -> backbone-native preprocessing (no double-rescaling — see below)
  -> EfficientNetB0 or ResNet50 convolutional backbone (ImageNet weights)
  -> GlobalAveragePooling2D
  -> Dropout(0.2)
  -> Dense(256, relu)
  -> BatchNormalization
  -> Dropout(0.3)
  -> Dense(1, sigmoid)        # flood probability in [0, 1]
```

### Two-Phase Progressive Fine-Tuning

| Phase | Trainable layers | Learning rate | Max epochs |
|-------|-----------------|---------------|------------|
| Phase 1 | Last 30 | 1e-4 | 15 |
| Phase 2 | All except first 50 | 1e-5 | 20 |

EarlyStopping (patience = 7) and ReduceLROnPlateau (factor = 0.5, patience = 3, min_lr = 1e-7) on validation loss. Class weights computed via `sklearn.utils.class_weight.compute_class_weight`.

### Hard Negative Mining Pipeline

1. **Confounder analysis** (`scripts/analyze_confounders.py`): predict on all `train/non_flood/` images; rank by flood probability; report per-category FP rates to identify mining targets.
2. **Mine**: flag training-set non-flood images with p(flood) > tau (default tau = 0.3, selected by validation sweep). Mining runs on `train/non_flood/` only — never on val or test.
3. **Augment**: apply 5× augmentation to each mined image (same transforms as training). Save to `processed_data/binary_hnm/augmented_hard_negatives/`.
4. **Inject**: copy original training set to `processed_data/binary_hnm/`; add augmented hard negatives to `non_flood/`. Val and test directories are never modified.
5. **Retrain**: Phase 1 (LR = 5e-5, 15 epochs), Phase 2 (LR = 1e-5, 10 epochs) on the augmented set. Validation monitored on the original unmodified val partition.
6. **Partition integrity assertion**: halts training if any filename overlap between train augmentations and val/test is detected.

### Extended-Training Control

`train_hnm.py --no_injection` runs the same epoch budget as the HNM model (Phase 1 + Phase 2 + HNM Phase 1 + HNM Phase 2) without injecting any hard negatives. This isolates the HNM contribution from the benefit of additional training epochs.

### Preprocessing

Backbone-native preprocessing is used via `tf.keras.applications.efficientnet.preprocess_input` / `resnet50.preprocess_input` with `include_preprocessing=False`. **Do not combine `ImageDataGenerator(rescale=1./255)` with `include_preprocessing=True`** — this double-rescales inputs to approximately [0, 0.004], rendering pretrained features non-functional. A `verify_preprocessing()` assertion runs before every training loop.

### Threshold Selection

Thresholds are swept from 0.05 to 0.95 on the **validation set** to find the operating point maximising flood recall subject to 0% swimming pool FP on validation. The test set is evaluated at the selected threshold exactly once per model configuration.

---

## Experimental Setup

### Ablations

| Ablation | Purpose |
|----------|---------|
| Extended training without HNM (`--no_injection`) | Isolates HNM contribution from extra training epochs — **run first** |
| Tau sweep (0.2, 0.3, 0.4, 0.5) | Select optimal mining threshold on validation |
| Augmentation factor (2×, 5×, 10×) | Sensitivity to augmentation multiplier |
| Phase 1 only / Phase 2 only / both | Contribution of each training phase |
| With/without class weights | Effect of class rebalancing |

### Statistical Validation

- 5-seed runs (42, 123, 256, 512, 1024): mean ± std reported for all metrics
- Bootstrap 95% CIs (1,000 resamples) on all test-set metrics
- McNemar's test for pairwise model accuracy comparisons
- Clopper-Pearson binomial CIs for swimming pool FP rate
- Fisher's exact test for pre/post-HNM pool FP comparison

### Visualisations

- Precision-Recall and ROC curves per model
- Confusion matrices with cell counts
- GradCAM++ heatmaps for FN, FP, swimming pool, and correctly classified sets (`scripts/grad_cam.py`)
- t-SNE / UMAP on Dense(256) embeddings, coloured by true label and swimming pool flag
- Reliability diagrams (pre- and post-calibration ECE)
- Severity-stratified recall bar chart

---

## Datasets and Evaluation

### Primary Dataset

**FloodingDataset2** (University of South Florida) — 3,754 street-level images.

```
FloodingDataset2/
  StreetFloodClasses/
    MajorFlood/    MinorFlood/    ModerateFlood/
    NoFlood/       parks_walkways/
  junk/
    Swimmingpool.zip, Cars.zip, Dogs.zip, ...  (12 distractor categories)
  processed_data/
    binary/
      train/  (flood/, non_flood/)   — 2,627 images  (70%)
      val/    (flood/, non_flood/)   —   563 images  (15%)
      test/   (flood/, non_flood/)   —   564 images  (15%)
```

Splits are stratified by `multiclass_label × is_swimming_pool` (seed 42). Hard negatives: 15 swimming pool images in test.

### Cross-Dataset Validation (planned)

- **CrisisMMD / CrisisBench** — social media ground-level disaster images; best domain match for generalisation testing.
- **AlleyFloodNet** — street-level alley flooding; directly comparable deployment scenario.
- Zero-shot transfer and lightweight head-only adaptation both reported.

### Evaluation Metrics

| Metric | Role |
|--------|------|
| **PR-AUC** | Primary discrimination metric |
| Recall | Primary operational metric (high recall ≥ 95% target) |
| F1, Precision, Accuracy | Secondary |
| ROC-AUC | Reported for comparison only |
| Pool FP rate + Clopper-Pearson CI | Confounder-specific FP analysis |
| ECE, reliability diagram | Probability calibration quality |
| Severity-stratified recall | Geoscience contribution |

---

## Pipeline

| Step | Notebook | Script | Compute |
|------|----------|--------|---------|
| 1. Data exploration | `01_data_exploration.ipynb` | — | CPU |
| 2. Stratified splitting | `02_stratified_splitting.ipynb` | — | CPU |
| 3. Baseline EfficientNetB0 | `03_baseline_efficientnetb0.ipynb` | `train_baseline.py --arch efficientnet` | T4 GPU |
| 4. Baseline ResNet50 | `04_baseline_resnet50.ipynb` | `train_baseline.py --arch resnet50` | T4 GPU |
| 5a. Confounder analysis | `05a_confounder_analysis.ipynb` | `analyze_confounders.py` | T4 GPU |
| 5b. HNM — EfficientNetB0 | `05b_hnm_efficientnetb0.ipynb` | `train_hnm.py --arch efficientnet` | T4 GPU |
| 5c. HNM — ResNet50 | `05c_hnm_resnet50.ipynb` | `train_hnm.py --arch resnet50` | T4 GPU |
| 6. Evaluation | `06_evaluation.ipynb` | `evaluate.py` | CPU/GPU |

**Execution order:** `03 → 04 → 05a → 05b → 05c → 06`

After step 05a, copy the printed checkpoint path into `MODEL_PATH` in the HNM notebooks before running.

---

## Ideal Results

The best-case outcome demonstrates three things:

1. **HNM beats the extended-training control.** If the `--no_injection` ablation achieves similar accuracy to HNM, the improvement is attributable to training budget, not mining. If HNM exceeds it — particularly on pool FP rate and minority-flood recall — the contribution is established.
2. **Cross-seed consistency.** Hard negatives identified by different random seeds substantially overlap, suggesting the mining step targets a stable region of the decision boundary rather than reflecting initialisation noise.
3. **PR-AUC ≥ 0.97 with ≥ 95% recall and ≤ 5% pool FP rate**, with 95% CIs that do not overlap the baseline. This would support the claim of a confounder-robust first-pass screening system.

---

## Known Limitations

1. **Dataset size and source diversity.** 3,754 images from a single institutional source is small for deep learning. Geographic and photographer diversity is not documented.
2. **Random splitting vs. event-based splitting.** If images from the same flood event appear in both train and test, the model may recognise the scene rather than the flood. Without event metadata this cannot be fully mitigated.
3. **Swimming pool sample size.** 15 pool images cannot support a statistically meaningful FP rate. A 0% observed rate has a 95% CI of approximately [0%, 21.8%]. Claims of "complete elimination" must not appear without this CI.
4. **Only 2 hard negatives mined in the CIBB baseline.** Whether the method generalises to other confounders (wet roads, irrigation channels, rain-obscured scenes) remains untested.
5. **Validation set double-duty in HNM pipeline.** The same 563-image validation set selects the baseline checkpoint used for mining and then selects the retrained HNM checkpoint. This introduces a weak dependency between the validation partition and the HNM training data. Test-set results are the definitive evaluation.
6. **Geoscience framing.** This is a first-stage filter within a larger geoscience pipeline, not a standalone hydrological analysis system. The contribution must be connected to operational flood monitoring infrastructure to meet *Computers & Geosciences* scope.

---

## Setup

### Local (CPU tasks)

```bash
pip install -r requirements.txt
```

### Colab (GPU training)

```python
from google.colab import drive
drive.mount('/content/drive')

%pip install -q tf-keras-vis umap-learn

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
  scripts/
    utils.py               # seeding, model building, preprocessing, callbacks
    train_baseline.py      # two-phase progressive fine-tuning
    analyze_confounders.py # rank train/non_flood categories by FP rate
    train_hnm.py           # HNM mining, augmentation injection, retraining
    evaluate.py            # PR-AUC primary, bootstrap CI, McNemar, severity recall
    grad_cam.py            # GradCAM++ heatmaps for FN/FP/pool/correct sets
  results/
    figures/          PR curves, confusion matrices, GradCAM grids, reliability diagrams
    tables/           Metric CSVs, confounder FP rates, tau sweep results
    logs/             Training history CSVs
    predictions/      Per-image prediction CSVs
  methodology.mmd     Pipeline diagram (Mermaid)
  requirements.txt
```

---

## License

MIT

## Citation

> [To be added upon acceptance]
