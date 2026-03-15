# Design: HNM Pipeline Fix + Comprehensive Evaluation
**Date:** 2026-03-15
**Project:** First-Pass Flood Binary Detection from Crowdsourced Imagery: Quality Control for Screening Systems
**Target venue:** Computers & Geosciences

---

## Context

The existing codebase has three scripts: `train_efficientnet.py` (baseline + HNM for EfficientNetB0), `train_resnet50.py` (baseline only), and `evaluate_model.py` (basic confusion matrix + classification report). Three critical problems block journal submission:

1. **Preprocessing mismatch (critical bug):** All scripts use `rescale=1./255` in `ImageDataGenerator` AND use backbone models with `include_preprocessing=True` (default in TF 2.16+). This double-rescaling means training sees inputs at `[0, ~0.004]` but mining inference feeds `[0, 1]` via manual `/255`. HNM confidence scores are unreliable because the model scores mining images on a completely different input distribution than it was trained on.

2. **ResNet50 has no HNM:** `train_resnet50.py` trains baseline only. The architectural comparison between EfficientNetB0+HNM and ResNet50 (no HNM) is confounded — it compares training budget + data, not architecture.

3. **Evaluation is insufficient for journal submission:** Missing PR-AUC (primary metric for imbalanced screening), bootstrap confidence intervals, GradCAM interpretability, per-confounder FP rates, McNemar's test, severity-stratified recall, and data-driven justification for the swimming pool HNM choice.

### Research gaps addressed

| Gap | Description | How addressed |
|-----|-------------|---------------|
| Gap 1 | No HNM for street-level flood classification | `analyze_confounders.py` + `train_hnm.py` with data-driven confounder selection |
| Gap 2 | No two-phase progressive fine-tuning ablation | `train_baseline.py --phase_boundary` param |
| Gap 3 | No per-confounder FP rates with CI | `analyze_confounders.py` + `evaluate.py` binomial CI per category |
| Gap 4 | PR-AUC not used as primary metric | `evaluate.py` — PR-AUC is headline metric, ROC-AUC secondary (note: probability calibration dropped as out of scope for this submission, acknowledged as future work) |
| Gap 5 | Confounded architectural comparisons | `train_hnm.py --no_injection` extended-training control |
| Gap 6 | No bootstrap CI or McNemar tests | `evaluate.py` 1000-resample bootstrap + McNemar between model pairs |

> **Note on gap numbering:** Gap 4 in the literature review was probability calibration (ECE, reliability diagrams). This is intentionally deferred — for a binary screening system with validation-set threshold tuning, post-hoc calibration is not a core contribution of this paper. It is acknowledged as future work in the limitations section.

### Goal framing

**Maximise recall subject to a false-positive rate constraint.** This corresponds to operating at a specific point on the Precision-Recall curve. PR-AUC captures this tradeoff across all operating points and is the recommended discrimination metric for imbalanced binary detection (Davis & Goadrich, 2006; Saito & Rehmsmeier, 2015). ROC-AUC is reported as a secondary metric only, explicitly noted as potentially optimistic under class imbalance.

---

## File Structure

### Scripts (source of truth)

```
scripts/
  utils.py                  # shared: seeding, model building, preprocessing, callbacks, data loaders
  train_baseline.py         # --arch efficientnet|resnet50 → Phase 1 + Phase 2 baseline
  analyze_confounders.py    # rank train/non_flood categories by FP rate → mining_candidates.txt
  train_hnm.py              # --arch --tau_mode percentile|sweep --no_injection
  evaluate.py               # full eval: PR-AUC primary, bootstrap CI, McNemar
  grad_cam.py               # GradCAM++ heatmaps on FN / FP / pool / correct sample images
```

### Notebooks (thin Colab wrappers — GPU via VS Code Colab extension)

```
notebooks/
  03_baseline_efficientnetb0.ipynb    # !python scripts/train_baseline.py --arch efficientnet
  04_baseline_resnet50.ipynb          # !python scripts/train_baseline.py --arch resnet50
  05a_confounder_analysis.ipynb       # !python scripts/analyze_confounders.py ...
  05b_hnm_efficientnetb0.ipynb        # !python scripts/train_hnm.py --arch efficientnet ...
  05c_hnm_resnet50.ipynb              # !python scripts/train_hnm.py --arch resnet50 ...
  06_evaluation.ipynb                 # !python scripts/evaluate.py + inline plots
```

### Files removed / replaced

| Old file | Replaced by |
|----------|-------------|
| `train_efficientnet.py` | `scripts/train_baseline.py` + `scripts/train_hnm.py` |
| `train_resnet50.py` | `scripts/train_baseline.py --arch resnet50` |
| `evaluate_model.py` | `scripts/evaluate.py` |

---

## Component Design

### utils.py

Shared utilities consumed by all scripts.

**Seeding (all four sources):**
```python
def set_all_seeds(seed: int) -> None:
    import random, os
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
```
Called at the top of every script before any data loading or model construction.

**Preprocessing (fixes the critical bug):**
```python
PREPROCESS_FN = {
    "efficientnet": tf.keras.applications.efficientnet.preprocess_input,
    "resnet50":     tf.keras.applications.resnet50.preprocess_input,
}
```
> **Design note:** This implements CLAUDE.md Option B (explicit preprocessing via `preprocessing_function`, backbone `include_preprocessing=False`). CLAUDE.md lists Option A (`include_preprocessing=True`, no generator rescale) as the recommendation, but Option B is equally correct and makes preprocessing behaviour explicit and verifiable. This is a deliberate choice, not an oversight.

- `ImageDataGenerator(preprocessing_function=PREPROCESS_FN[arch])` — **no `rescale` param anywhere**
- Backbone loaded with `include_preprocessing=False`
- **Mining inference must also call `PREPROCESS_FN[arch]` on each image loaded individually — never use manual `/255.0` rescaling.** This is the root cause of the original bug and must be enforced explicitly wherever images are loaded outside a generator.
- Applies identically to: training generators, HNM mining inference, evaluation generators

**Architecture-specific preprocessing verification (run before every training run):**
```python
def verify_preprocessing(generator, arch: str) -> None:
    batch, _ = next(generator)
    if arch == "resnet50":
        # resnet50.preprocess_input mean-subtracts; values go very negative
        assert batch.min() < -50, (
            f"ResNet50 generator output min={batch.min():.1f}; expected < -50. "
            "Check that preprocessing_function=resnet50.preprocess_input and rescale is not set."
        )
    elif arch == "efficientnet":
        # efficientnet.preprocess_input maps to [-1, 1]
        assert batch.min() < 0 and batch.max() <= 1.01, (
            f"EfficientNetB0 generator output range=[{batch.min():.3f}, {batch.max():.3f}]; "
            "expected [-1, 1]. Check preprocessing_function and rescale."
        )
    print(f"[OK] Preprocessing verified for {arch}: range=[{batch.min():.3f}, {batch.max():.3f}]")
```

**Model building:**
- `build_model(arch, phase_boundary=(30, 50))` — builds EfficientNetB0 or ResNet50 with the standard head (GAP → Dropout(0.2) → Dense(256, relu) → BN → Dropout(0.3) → Dense(1, sigmoid))
- `include_preprocessing=False` on both backbones
- `freeze_for_phase1(base, n_trainable=30)` — freeze all but last N layers
- `freeze_for_phase2(base, n_frozen=50)` — freeze first N layers only

**Class label mapping (explicit, not assumed):**
The label mapping from `flow_from_directory` with `class_mode="binary"` is alphabetical: `flood=0, non_flood=1`. **Hard negatives are non-flood images (label=1) that the model predicts as flood with high probability.** The mining condition is therefore:
```python
# label == 1 means non_flood
# pred is model output = P(non_flood) if trained with binary labels flood=0, non_flood=1
# flood probability = 1 - pred
if label == 1 and (1.0 - pred) > threshold:
    hard_negatives.append(path)
```
> **Bug note:** The existing `train_efficientnet.py` has this inverted — it gates on `label != 1` which processes flood images (label=0) instead of non-flood images (label=1). The `1.0 - pred` formula for flood probability is **correct** and must be kept. Only the gating condition needs to change: `label != 1` → `label == 1`. Do not change the probability formula.

**Shared callbacks:**
- `build_callbacks(checkpoint_path, patience, lr_patience, log_path)` — ModelCheckpoint (monitor=val_loss), EarlyStopping (patience), ReduceLROnPlateau (factor=0.5, min_lr=1e-7), CSVLogger(log_path)
- Phase 1 patience=7, Phase 2 patience=7 (consistent with CLAUDE.md hyperparameter table; deviations must be explicitly justified in the training script)

---

### train_baseline.py

**Args:** `--arch`, `--data_dir`, `--output_dir`, `--phase_boundary` (default `30,50`), `--seed` (default 42)

**Flow:**
1. `set_all_seeds(seed)`
2. Build generators with `PREPROCESS_FN[arch]` — no rescale
3. Call `verify_preprocessing(train_gen, arch)` — halt if assertion fails
4. Build model via `utils.build_model(arch)`
5. Phase 1: last 30 layers trainable, LR=1e-4, max 15 epochs, EarlyStopping patience=7
6. Phase 2: freeze first 50 layers, LR=1e-5, max 20 epochs, EarlyStopping patience=7
7. Save best checkpoint for each phase + final model to `--output_dir`
8. Training history saved via CSVLogger to `results/logs/{arch}_baseline_{timestamp}.csv`

**Both models run through identical protocol** — the only difference is the backbone and its preprocessing function.

---

### analyze_confounders.py

**Args:** `--model_path`, `--arch`, `--data_dir`, `--fp_threshold` (default 0.15), `--output_dir`

**Purpose:** Data-driven justification for which categories to mine as hard negatives.

**IMPORTANT — partition safety:** This script scans images from `processed_data/binary/train/non_flood/` only, grouped by category using filename prefix or the split manifest CSV. It must NOT scan the raw `StreetFloodClasses/` or `junk/extracted/` directories, as those contain images that may have ended up in val or test splits.

**Flow:**
1. Load trained baseline model; apply `PREPROCESS_FN[arch]` for inference (not manual rescaling)
2. For each non-flood category in `train/non_flood/`:
   - Groups: filename prefix identifies category (e.g. `Swimmingpool_001.jpg`, `NoFlood_042.jpg`)
   - If filenames do not retain category prefix, use the split manifest CSV (`train_split.csv`) which records `category` per image
3. Run inference on every image in each category group
4. Compute FP rate (% where `1 - pred > 0.5`) per category
   > **Design note:** The 0.5 decision boundary here is used solely for **ranking categories** by confusability. It is independent of the mining tau used in `train_hnm.py`. Changing the mining tau (e.g. sweep over 0.3–0.7) does not retroactively change which categories are flagged as mining candidates. The category flagging threshold (`--fp_threshold`, default 0.15) and the per-image mining tau are separate parameters serving different purposes.
5. Sort by FP rate descending, save to `results/tables/confounder_fp_rates_{arch}.csv`
6. Flag categories with FP rate > `--fp_threshold` as mining candidates
7. Write `results/mining_candidates_{arch}.txt` — one absolute directory path per flagged category's training images
8. Print ranked table to stdout

**Output narrative:** "We selected hard negative mining candidates by running the baseline model across all non-flood categories in the training partition. Categories with FP rate exceeding 15% were flagged as confounders [table]. This data-driven selection identified Swimmingpool [and others if applicable] as the primary sources of false alarms, providing a principled justification for the HNM target categories."

---

### train_hnm.py

**Args:** `--arch`, `--model_path` (baseline Phase 2 checkpoint), `--data_dir`, `--output_dir`, `--tau_mode` (percentile|sweep), `--top_pct` (default 0.10), `--no_injection`, `--phase_boundary` (default `30,50` — must match the value used in `train_baseline.py` for this run)

**Mining pool:** reads `results/mining_candidates_{arch}.txt` — mines only from flagged categories in the training partition.

**Partition safety assertion (halts on violation, no try/except):**
```python
leaked = set(mining_files) & (set(val_files) | set(test_files))
assert len(leaked) == 0, f"LEAKAGE: {len(leaked)} mining files overlap val/test. Halting."
```

**Mining inference preprocessing:** Images loaded individually must use `PREPROCESS_FN[arch]`:
```python
img = load_img(path, target_size=IMG_SIZE)
x = img_to_array(img)  # [0, 255] range — no /255.0
x = PREPROCESS_FN[arch](x[None, ...])  # apply backbone-specific preprocessing
pred = model.predict(x, verbose=0)[0][0]
```

#### tau_mode=percentile (default, fast)
1. Score all images in mining candidate directories using `PREPROCESS_FN[arch]`
2. Rank by flood probability (`1 - pred`) descending
3. Take top `--top_pct` (10%) as hard negatives
4. Augment 5× (rotation, flip, zoom, brightness — same transforms as training), save to `binary_hnm/augmented/`
5. Copy original training set + inject augmented hard negatives into `binary_hnm/train/`
6. HNM Phase 1: LR=5e-5, last 30 layers trainable, max 15 epochs, patience=7, monitor val from original unmodified `binary/val/`
7. HNM Phase 2: LR=1e-5, freeze first 50 layers, max 10 epochs, patience=7

#### tau_mode=sweep (ablation)
1. Sweep `tau ∈ {0.3, 0.4, 0.5, 0.6, 0.7}`
2. For each tau: mine images with `(1 - pred) > tau`, augment 5×, inject into training set copy
3. **Retrain head only** (backbone fully frozen) on **training data** for 5 epochs, evaluating on validation set after each epoch — val set is evaluated on, NOT trained on
4. Record val Recall and val PR-AUC at epoch 5 for each tau
5. Pick best tau by val Recall (primary), then val PR-AUC (tiebreaker)
6. Retrain full model with winning tau using full HNM Phase 1 + 2 schedule (same as percentile mode)
7. Save sweep results to `results/tables/tau_sweep_{arch}.csv`

> **Known limitation (acknowledge in paper):** The tau sweep uses the validation set both to select the best tau and (via early stopping in the full retrain) to select the best checkpoint. This double-use of the validation set is a weak leakage vector — the same partition influences both tau selection and model selection. With only 563 validation images, a dedicated tau-selection split is not practical. This must be acknowledged in the paper's limitations section (analogous to the "validation set double-duty in HNM pipeline" limitation already documented in CLAUDE.md).

#### --no_injection (extended-training control, Gap 5)
- **Loads the same baseline Phase 2 checkpoint** passed via `--model_path` (same starting weights as HNM)
- Trains for the **maximum epoch budget** of the HNM path (HNM Phase 1 max + HNM Phase 2 max = 25 epochs) on the **original training set with no hard negatives injected**
- Uses max budget rather than actual-epochs-run to avoid coupling; early stopping still applies
- Isolates whether the performance gain from HNM comes from the hard negatives themselves vs. simply more training time
- Results saved as `{arch}_extended_baseline_{timestamp}.keras`

---

### evaluate.py

**Args:** `--model_path`, `--arch`, `--data_dir`, `--output_dir`, `--n_bootstrap` (default 1000), `--compare_predictions_csv` (optional, for McNemar)

**Primary metric: PR-AUC** — headline number in all output tables. Output ordered as:

```
PR-AUC (primary) | Recall | Precision | F1 | ROC-AUC* | Accuracy
* ROC-AUC: secondary metric — see Davis & Goadrich (2006) re: limitations under class imbalance
```

**Metrics computed:**
- Accuracy, Precision, Recall, F1
- **PR-AUC** (`sklearn.metrics.average_precision_score`) — primary
- ROC-AUC (secondary)
- Confusion matrix with cell counts
- Full classification report

**Bootstrap CI (1000 resamples) on PR-AUC, Recall, ROC-AUC, F1, pool FP rate:**
```python
for _ in range(n_bootstrap):
    idx = np.random.choice(len(y_true), len(y_true), replace=True)
    # compute metric on resampled idx
# CI = [2.5th percentile, 97.5th percentile] across bootstrap samples
```

**Per-confounder FP analysis:**
- Swimming pool FP rate with binomial 95% CI (Clopper-Pearson)
- Fisher exact test (one-sided) vs baseline pool FP rate — output must include the significance conclusion explicitly: e.g. "p=0.11 — NOT significant at alpha=0.05" (per CLAUDE.md honest-reporting requirement)
- FP rate for any other flagged categories from `mining_candidates.txt`

**McNemar's test:**
- Requires `--compare_predictions_csv` pointing to a second model's saved predictions CSV
- Pairwise comparisons: baseline vs HNM, EfficientNetB0 vs ResNet50, HNM vs extended-baseline-no-injection
- With Bonferroni correction (divide alpha by number of comparisons)
- If `--compare_predictions_csv` not provided, McNemar section is skipped with a warning

**Severity-stratified recall:**
- Parses category from split manifest CSV (`test_split.csv`, column `category`) — **does not rely on filename parsing** since filenames may not retain category prefix after copying
- Reports recall per flood severity: MajorFlood, ModerateFlood, MinorFlood
- If manifest CSV unavailable, this section is skipped with a warning

**Outputs:**
- `results/tables/{model_name}_metrics.csv`
- `results/predictions/{model_name}_predictions.csv` — columns: filename, true_label, predicted_label, probability, correct, category
- `results/figures/{model_name}_pr_curve.png` (primary figure — PR curve with operating point marked)
- `results/figures/{model_name}_roc_curve.png`
- `results/figures/{model_name}_confusion_matrix.png`
- `results/figures/{model_name}_severity_recall.png` (if manifest available)

---

### grad_cam.py

**Args:** `--model_path`, `--arch`, `--predictions_csv`, `--data_dir`, `--output_dir`, `--n_per_set` (default 10)

**GradCAM variant: GradCAM++** (via `tf-keras-vis`). Preferred over vanilla GradCAM for flood detection because it handles multiple activation peaks per class better — relevant when the model attends to scattered water features rather than a single region.

**Layer selection (auto per arch):**
- EfficientNetB0: `top_conv`
- ResNet50: `conv5_block3_out`

**Image sets (loaded from `--predictions_csv`):**
1. **False negatives** — floods the model missed (highest operational cost for a screening system)
2. **False positives** — non-floods predicted as flood (false alarms)
3. **Swimming pool images** — verify model attends to water features; check whether attention shifts pre/post HNM
4. **Random correct predictions** — baseline visual comparison (5 flood correct + 5 non-flood correct)

**Preprocessing for GradCAM:** images must be preprocessed with `PREPROCESS_FN[arch]`. The required loading pattern is:
```python
img = load_img(path, target_size=IMG_SIZE)
x = img_to_array(img)             # [0, 255] — no /255.0 division
x = PREPROCESS_FN[arch](x[None, ...])   # backbone-specific preprocessing
```
This matches the mining inference pattern in `train_hnm.py`. Any division by 255 here would silently reintroduce the original preprocessing bug.

**Output:** 4 heatmap grid PNGs saved to `results/figures/gradcam_{arch}_{set}.png`

---

## Both Models — Full Pipeline

Every script is parameterized by `--arch`. Both EfficientNetB0 and ResNet50 run through the identical pipeline:

| Step | EfficientNetB0 | ResNet50 |
|------|----------------|----------|
| Baseline | `train_baseline.py --arch efficientnet` | `train_baseline.py --arch resnet50` |
| Confounder analysis | `analyze_confounders.py --arch efficientnet` | `analyze_confounders.py --arch resnet50` |
| HNM percentile | `train_hnm.py --arch efficientnet --tau_mode percentile` | `train_hnm.py --arch resnet50 --tau_mode percentile` |
| HNM sweep | `train_hnm.py --arch efficientnet --tau_mode sweep` | `train_hnm.py --arch resnet50 --tau_mode sweep` |
| Extended control | `train_hnm.py --arch efficientnet --no_injection` | `train_hnm.py --arch resnet50 --no_injection` |
| Evaluate | `evaluate.py --arch efficientnet --model_path ...` | `evaluate.py --arch resnet50 --model_path ...` |
| McNemar | `evaluate.py ... --compare_predictions_csv other_model.csv` | same |
| GradCAM | `grad_cam.py --arch efficientnet` | `grad_cam.py --arch resnet50` |

---

## Notebook Structure (Colab via VS Code Extension)

Each notebook follows this pattern:
1. Mount Google Drive
2. Environment check (`!nvidia-smi`, TF version)
3. `!python scripts/{script}.py --arg1 val1 ...`
4. Inline display of saved figures from `results/figures/`
5. Checkpoint save assertion to Drive

Notebooks are intentionally thin — all logic lives in scripts. This keeps version control clean and allows scripts to be re-run outside Colab if needed.

---

## Verification

### End-to-end test sequence
1. Run `train_baseline.py --arch efficientnet` — verify checkpoint saved, training CSV written, preprocessing assertion passes
2. Run `train_baseline.py --arch resnet50` — same
3. Run `analyze_confounders.py --arch efficientnet` — verify ranked FP table, `mining_candidates_efficientnet.txt` non-empty, categories are from `train/non_flood/` only
4. Run `train_hnm.py --arch efficientnet --tau_mode percentile` — verify partition assertion passes, hard negatives found > 0, HNM checkpoint saved
5. Run `train_hnm.py --arch efficientnet --tau_mode sweep` — verify `tau_sweep_efficientnet.csv` written, best tau selected and used for full retrain
6. Run `train_hnm.py --arch efficientnet --no_injection` — verify same starting checkpoint as HNM, same total epochs, no augmented images in `binary_hnm/`
7. Run `evaluate.py --arch efficientnet` on all model variants — verify PR-AUC is headline metric, bootstrap CI columns present, predictions CSV saved
8. Run `evaluate.py` with `--compare_predictions_csv` — verify McNemar output present
9. Run `grad_cam.py --arch efficientnet` — verify 4 heatmap grids saved
10. Repeat steps 1-9 for `--arch resnet50`

### Preprocessing sanity check (utils.py, called before every training run)
Architecture-specific — see `verify_preprocessing()` in utils.py section above.

### Partition integrity check (train_hnm.py, runs before mining)
```python
leaked = set(mining_files) & (set(val_files) | set(test_files))
assert len(leaked) == 0, f"LEAKAGE: {len(leaked)} mining files overlap val/test. Halting."

# Also assert augmented files are not in val/test (CLAUDE.md reproducibility checklist)
augmented_files = set(os.listdir(hnm_augmented_dir)) if os.path.exists(hnm_augmented_dir) else set()
held_out = set(val_files) | set(test_files)
assert len(augmented_files & held_out) == 0, "Augmented HNM files found in val/test. Halting."
```
No try/except. Hard failure only.

### evaluate.py and grad_cam.py partition check
Both scripts load test images. Each must verify at startup that the test directory path points to `processed_data/binary/test/`, not to `processed_data/binary_hnm/` or any other modified directory:
```python
assert "binary_hnm" not in str(test_dir), (
    f"evaluate.py must load from the original unmodified test partition, got: {test_dir}"
)
assert os.path.exists(test_dir), f"Test directory not found: {test_dir}"
```
