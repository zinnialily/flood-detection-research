# CLAUDE.md -- Flood Detection Study (Computers & Geosciences Submission)

## Project Overview


## THERE IS AN EMPHASIS ON HIGH RECALL BC U DONT WANNA REMOVE A TRUE ONE

This project develops a binary flood detection system from street-level imagery using hard
negative mining (HNM) with progressive transfer learning. The target journal is **Computers &
Geosciences** (Elsevier, ISSN 0098-3004), which publishes high-impact original research at the
interface of computer science and geosciences. The journal requires open-source code, rigorous
reproducibility, and a clear geoscience contribution -- not just a machine learning exercise.

**Repository:** <https://github.com/CRIS-Hazard/imagevalidation> (reference only; all new
work lives in this workspace).

**Dataset (Google Drive):**
<https://drive.google.com/drive/folders/1PXc9VTeQgV5WeNxa2NOC9kbRpnQtg20o>

---

## Critical Priority: Experiment Execution Order

The experiments described in this document are numerous and resource-intensive. However, not
all carry equal weight. The following execution order reflects methodological dependencies: if
an early experiment invalidates the core claims, later experiments become unnecessary or must
be redesigned.

**Phase A -- Foundational corrections (must complete before any other work):**

1. **Fix preprocessing and retrain all three models** (Steps 3, 4, 5). The double-rescaling
   bug invalidates every existing result. No other experiment is meaningful until this is
   resolved.
2. **Run the extended-training-without-HNM ablation** (Step 8, "Extended training without
   HNM" row). This is the single most important experiment in the pipeline. If the
   baseline model, given the same total training budget as the HNM model, matches or
   nearly matches HNM performance, the paper's central claim collapses and must be
   reframed. Run this BEFORE investing in Steps 7-17.
3. **Add ResNet50+HNM** (Step 6 addition). The current design does not support the
   architectural interaction claim because HNM was only applied to EfficientNetB0. Either
   add this experiment or remove the claim from the paper entirely.

**Phase B -- Statistical robustness (required for any journal submission):**

4. Multi-seed runs (Step 7).
5. Core ablation studies (Step 8).

**Phase C -- Extended experiments (required for Computers & Geosciences scope):**

6. Steps 9-17 in their documented order.

**Decision gate after Phase A:** If the extended-training ablation closes most of the gap
between baseline and HNM, the paper must be reframed. The contribution becomes the
two-phase progressive fine-tuning protocol and the methodological framework for HNM in
flood detection, not the HNM accuracy improvement itself. Plan for this possibility.

---

## Compute Environment: Google Colab via VS Code Extension

All GPU-dependent work (training, inference, Grad-CAM, large-batch evaluation) MUST run on
Google Colab's free T4 GPU through the official VS Code extension. CPU-only tasks (data
organisation, plotting, metric computation, file management) may run locally.

### How the Colab + VS Code Setup Works

1. **Open or create a `.ipynb` notebook** in VS Code.
2. **Install the Google Colab extension** from the VS Code marketplace (publisher: Google).
3. **Switch the kernel** to "Colab" via the kernel picker in the top-right of the notebook
   editor. Select **"New Colab Server"** and choose the **T4 GPU** accelerator.
4. **Wait for the connection.** Running `!nvidia-smi` should report `Tesla T4`. If it still
   shows a local GPU or CPU, the kernel has not switched -- retry.
5. **Mount Google Drive** in every Colab session:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
6. The dataset lives at `/content/drive/MyDrive/FloodingDataset2/` once mounted.

### Session Constraints

| Constraint              | Value                                      |
|-------------------------|--------------------------------------------|
| Max continuous runtime  | ~12 hours (free tier)                      |
| GPU type                | Tesla T4 (16 GB VRAM)                      |
| RAM                     | ~12 GB (free tier)                         |
| Disk                    | ~78 GB (Colab VM)                          |
| Idle timeout            | ~90 minutes of inactivity                  |
| Daily GPU quota         | Variable; may be throttled after heavy use |

### Rules for Working Within Colab Limits

- **Always checkpoint aggressively.** Save model weights to Google Drive after every phase
  and after every epoch via `ModelCheckpoint`. Never rely on Colab VM disk alone.
- **Use `CSVLogger`** for all training runs so logs survive disconnections.
- **Batch long experiments.** If a run will exceed 10 hours, split it into resumable phases
  (Phase 1 training, Phase 2 fine-tuning, HNM retraining) in separate notebooks.
- **Keep the session alive.** When running long training jobs, keep the VS Code window
  focused or use a keep-alive cell (a simple JS interval) in a secondary notebook tab.
- **Fall back to CPU for non-training tasks.** Data exploration, plotting, CSV analysis, and
  report generation do not need a GPU. Run these locally or on a Colab CPU runtime to
  preserve GPU quota.
- **Alternative free GPU sources** (use only if Colab quota is exhausted):
  - Kaggle Notebooks: 30 hours/week of T4 GPU.
  - Lightning AI Studios: free tier with limited GPU hours.
  - Google Cloud free trial: $300 credit for 90 days (requires credit card).

---

## Dataset

**Location on Drive:** `/content/drive/MyDrive/FloodingDataset2/`

```
FloodingDataset2/
  StreetFloodClasses/
    MajorFlood/
    MinorFlood/
    ModerateFlood/
    NoFlood/
    parks_walkways/
  junk/
    Cats.zip, Deers.zip, Dogs.zip, Motorcycle.zip, Plants.zip,
    Swimmingpool.zip, building_exterior.zip, building_interior.zip,
    bus.zip, car.zip, house_exterior.zip, Racoon.zip
  processed_data/
    binary/
      train/  (flood/, non_flood/)
      val/    (flood/, non_flood/)
      test/   (flood/, non_flood/)
```

| Split | Images | Percentage |
|-------|--------|------------|
| Train | 2,627  | 70%        |
| Val   | 563    | 15%        |
| Test  | 564    | 15%        |

- **Flood categories:** MajorFlood (48.2% of test flood), MinorFlood (32.1%), ModerateFlood (19.7%).
- **Non-flood categories:** NoFlood, parks_walkways, plus 12 junk/distractor categories.
- **Hard negatives:** Swimmingpool (15 images in test set).
- **Stratification key:** `multiclass_label + "_" + is_swimming_pool`, seed 42.
- **Image input size:** 224 x 224, rescaled to [0, 1].

---

## Study Pipeline (Execution Order)

Each step below corresponds to a notebook. Name notebooks with a numeric prefix and
descriptive slug: `01_data_exploration.ipynb`, `02_stratified_splitting.ipynb`, etc.

### Step 1 -- Data Exploration and Preparation

**Notebook:** `01_data_exploration.ipynb`
**Compute:** CPU (local or Colab CPU runtime).

- Mount Drive. Unzip all `.zip` files in `junk/` to `extracted/junk/`.
- Audit every category: count images, check for corruption, record dimensions and formats.
- Generate class distribution bar chart, imbalance visualisation, image size scatter plot,
  and a sample image grid (5 images per category).
- Save `dataset_statistics.csv` and `dataset_report.txt`.

### Step 2 -- Stratified Splitting

**Notebook:** `02_stratified_splitting.ipynb`
**Compute:** CPU.

- Build a DataFrame of all image paths with columns: `path`, `category`, `binary_label`,
  `multiclass_label`, `is_swimming_pool`.
- Create `stratify_key = multiclass_label + "_" + str(is_swimming_pool)`.
- Split 70/15/15 using `train_test_split` with `random_state=42`.
- Copy files into `processed_data/binary/{train,val,test}/{flood,non_flood}/`.
- Verify proportional representation of flood severity and swimming pool images per split.

#### Stratification Gap in Junk Categories (MUST BE VERIFIED)

The current stratification key is `multiclass_label + "_" + is_swimming_pool`. This ensures
proportional representation of flood severity levels and swimming pool images, but it does
NOT stratify by the 12 individual junk/distractor categories (Cats, Dogs, Motorcycle,
building_exterior, etc.). These categories vary in size and visual characteristics. A random
split could, by chance, concentrate certain junk categories in the training set, producing
a test set with a different non-flood composition than training.

The non-flood class conflates fundamentally different image domains: outdoor scenes (parks,
walkways), architectural imagery (building exterior/interior, house exterior), vehicles
(bus, car, motorcycle), animals (cats, dogs, deer, raccoon), and swimming pools. The
visual diversity within non-flood far exceeds that within flood.

**Required verification (add to this notebook):**

```python
# After splitting, verify per-category distribution across splits
for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    print(f"\n{split_name} ({len(split_df)} images):")
    for cat in sorted(split_df['category'].unique()):
        count = len(split_df[split_df['category'] == cat])
        pct = count / len(split_df) * 100
        print(f"  {cat}: {count} ({pct:.1f}%)")
```

If any source category has fewer than 5 images in any split, flag it as a potential source
of evaluation variance in the paper. Ideally, include the full source category in the
stratification key (`category + "_" + is_swimming_pool`), though this may fail for very
small categories where scikit-learn cannot place at least one sample per stratum in each
split. In that case, group small categories (e.g., all animal categories into "animals")
before stratification.

### Step 3 -- Baseline EfficientNetB0

**Notebook:** `03_baseline_efficientnetb0.ipynb`
**Compute:** Colab T4 GPU.

#### Preprocessing Bug Fix (CRITICAL -- BLOCKS ALL DOWNSTREAM WORK)

The original CIBB 2026 code has a double-rescaling bug that affects both EfficientNetB0
and ResNet50. The saved model summary confirms this layer sequence at the top of the
EfficientNetB0 model:

```
input_layer_1   (InputLayer)      receives output of ImageDataGenerator(rescale=1./255)
rescaling_2     (Rescaling)       EfficientNet built-in: scales by 1/255
normalization_1 (Normalization)   EfficientNet built-in: ImageNet mean/std
rescaling_3     (Rescaling)       EfficientNet built-in: final range mapping
stem_conv_pad   (ZeroPadding2D)   actual convolutions begin here
```

The three preprocessing layers (rescaling_2, normalization_1, rescaling_3) are all INTERNAL
to `EfficientNetB0` as shipped by Keras in TF 2.16+ when `include_preprocessing=True`
(the default). They expect raw [0, 255] pixel input.

The bug: `ImageDataGenerator(rescale=1./255)` scales pixels to [0, 1] BEFORE the model
receives them. Then the model's built-in rescaling_2 divides by 255 again, producing
values in [0, ~0.004]. The pretrained ImageNet weights expect inputs in [0, 1] after
their own rescaling -- they receive values ~255x too small.

The same issue applies to ResNet50 in TF 2.16+, which includes built-in preprocessing
that expects [0, 255] input.

#### Severity of the Bug: Transfer Learning Was Non-Functional

This bug is more severe than simple numerical compression. The pretrained ImageNet features
in the FROZEN early layers receive inputs approximately 255x smaller than expected. These
layers produce near-zero activations propagated through ReLU nonlinearities. This means the
frozen backbone was functioning essentially as a near-zero feature extractor during Phase 1.
The model learned almost entirely from the last 30 unfrozen layers, which were being trained
from near-scratch on a highly compressed input distribution.

**Consequence: the "transfer learning" narrative of the CIBB paper is fundamentally
misleading for the existing results.** The pretrained representations were not meaningfully
transferred. The 85.46% baseline EfficientNetB0 accuracy is the performance of a partially
random network trained on 2,627 images, not a fine-tuned ImageNet model.

This also means the comparative claims between architectures are invalid. EfficientNetB0
and ResNet50 respond differently to near-zero inputs due to their distinct internal
normalisation schemes (batch normalisation placement, squeeze-and-excitation blocks,
residual connections). The claim that "EfficientNetB0's compound scaling approach proves
more compatible with hard negative mining than ResNet50's residual hierarchy" cannot be
supported when both architectures were evaluated under a preprocessing regime that corrupts
their intended input distributions differently.

**Consequence: ALL experiments must be re-run with fixed preprocessing.** The existing
CIBB results (85.46% baseline, 94.15% ResNet50, 98.76% HNM) were trained with the
double-rescaling bug. Do not mix old and new checkpoints. Re-running everything with
correct preprocessing is a requirement for the journal submission anyway (multi-seed
runs, ablations, new baselines). Report only the corrected results in the paper. The
existing results should not appear in the journal paper even as a reference point,
because they could mislead readers into thinking the performance gap between corrected
and bugged models represents an effect size attributable to the methodology.

**The fix -- choose ONE of these two approaches (Option A is recommended):**

**Option A (recommended): Let the backbone handle preprocessing.**
Remove `rescale=1./255` from all ImageDataGenerators. Feed raw [0, 255] pixels. The
backbone's built-in layers handle rescaling and normalization correctly.

```python
# CORRECT: no rescale -- backbone handles it internally
train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    fill_mode='reflect',
    # NOTE: no rescale parameter
)
val_test_datagen = ImageDataGenerator()  # no rescale

base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    include_preprocessing=True,   # default; built-in rescaling + normalization
    input_shape=(224, 224, 3),
)
```

**Option B (explicit control): Disable built-in preprocessing, do it yourself.**
Use `include_preprocessing=False` and apply a single `Rescaling(1./255)` layer either
in the generator or in the model, not both.

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    # ... other augmentation params ...
)
val_test_datagen = ImageDataGenerator(rescale=1./255)

base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    include_preprocessing=False,  # disable built-in rescaling
    input_shape=(224, 224, 3),
)
```

**Verification cell (include in every training notebook):**
```python
# After building the model, verify input expectations
sample_batch, _ = next(train_generator)
print(f"Generator output range: [{sample_batch.min():.4f}, {sample_batch.max():.4f}]")

# If using Option A (include_preprocessing=True):
#   Generator should output [0, 255] range
#   If you see [0, 1], the generator is rescaling -- remove rescale param
# If using Option B (include_preprocessing=False):
#   Generator should output [0, 1] range
#   If you see [0, 255], add rescale=1./255 to the generator

# Also verify by checking model layers:
for layer in model.layers[:6]:
    print(f"  {layer.name:30s}  params={layer.count_params()}")
```

#### Architecture (corrected, for all EfficientNetB0 variants)

```
Input (224x224x3, raw pixels [0, 255])
  -> [EfficientNetB0 built-in preprocessing]
       -> Rescaling(1/255)           [0, 255] -> [0, 1]
       -> Normalization(ImageNet)    mean/std centering
       -> Rescaling(range adjust)    to backbone's expected range
  -> [EfficientNetB0 convolutional backbone, ImageNet weights]
  -> GlobalAveragePooling2D          1280-d feature vector
  -> Dropout(0.2)
  -> Dense(256, relu)
  -> BatchNormalization
  -> Dropout(0.3)
  -> Dense(1, sigmoid)               flood probability in [0, 1]
```

Training protocol:
- **Phase 1 (partial unfreeze):** Last 30 layers trainable. LR = 1e-4. Up to 15 epochs.
  EarlyStopping patience=7, ReduceLROnPlateau factor=0.5 patience=3 min_lr=1e-7.
- **Phase 2 (extended fine-tune):** Freeze only first 50 layers. LR = 1e-5. Up to 20 epochs.
  Same callbacks.
- **Class weights:** Computed via `sklearn.utils.class_weight.compute_class_weight`.
- **Augmentation (train only):** rotation_range=15, horizontal_flip=True,
  width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2,
  brightness_range=[0.8,1.2], fill_mode='reflect'. No rescale (Option A).
- **Batch size:** 32.

Save Phase 1 and Phase 2 best weights to Drive. Log to CSV and TensorBoard.

### Step 4 -- Baseline ResNet50

**Notebook:** `04_baseline_resnet50.ipynb`
**Compute:** Colab T4 GPU.

Identical protocol to Step 3, substituting `ResNet50` backbone (25.6M params vs 5.3M).
Same classification head, same augmentation, same two-phase schedule, same callbacks.
**Same preprocessing fix applies:** either remove `rescale=1./255` from generators and
use `ResNet50(include_preprocessing=True)`, or use `include_preprocessing=False` and
keep the generator rescale. Be consistent with the EfficientNetB0 choice across all
models so comparisons are fair.

#### ResNet50 Phase 2 Instability (OBSERVED IN EXISTING TRAINING LOGS)

The existing Phase 2 training logs for ResNet50 show severe instability: validation accuracy
drops from ~87% at the end of Phase 1 to ~55% at the start of Phase 2 (epoch 15), then
slowly recovers over 15 epochs. This is catastrophic forgetting caused by unfreezing too
many layers simultaneously. The transition from freezing all-but-30 layers to freezing
only the first 50 layers dramatically increases the number of trainable parameters, and the
Phase 2 learning rate (1e-5) is not sufficiently small to prevent destabilisation.

Under the double-rescaling bug this was especially severe because the frozen layers were
producing near-zero activations; unfreezing them introduced a sudden influx of gradient
signal into dormant layers. After fixing preprocessing, the instability may be reduced
because frozen layers will carry meaningful pretrained features. However, monitor the
Phase 1-to-Phase 2 transition carefully.

**If instability persists after preprocessing fix, consider these mitigations:**

1. **Gradual unfreezing:** Instead of unfreezing all layers past index 50 at once, unfreeze
   10 additional layers every 2 epochs during Phase 2.
2. **Discriminative learning rates:** Assign earlier layers a smaller LR (e.g., 1e-6) and
   later layers a larger LR (e.g., 1e-5). This prevents early pretrained features from
   being overwritten while allowing task-specific adaptation in later layers.
3. **Warmup schedule:** Start Phase 2 with LR = 1e-7 and linearly warm up to 1e-5 over
   the first 3 epochs.

### Step 5 -- Hard Negative Mining (EfficientNetB0)

**Notebook:** `05_hard_negative_mining.ipynb`
**Compute:** Colab T4 GPU.

#### HNM Data Leakage Prevention (CRITICAL)

Hard negative mining introduces a second pass over data before retraining, which creates
multiple opportunities for information from validation or test sets to contaminate the
training process. Every step below must enforce strict partition isolation.

**Iron rules -- violations invalidate all downstream results:**

- Mining predictions are computed ONLY on `processed_data/binary/train/non_flood/`.
  Never run the mining scan on val or test images.
- The confidence threshold tau MUST be selected BEFORE looking at test performance.
  Use validation-set metrics (or a held-out portion of the training set) to choose tau.
  The ablation study (Step 8) sweeps tau on validation data, not test data.
- Augmented hard negatives are injected ONLY into the training set. The val and test
  generators must load from their original, unmodified directories.
- The model checkpoint used to mine hard negatives is the Phase 2 best model selected
  by validation loss during baseline training (Step 3). It was never exposed to test data.
- After retraining, final evaluation on the test set happens exactly once per model
  configuration. Do not iterate on the test set to tune any hyperparameter.

#### Validation Set Double-Duty Leakage Vector (ACKNOWLEDGE IN PAPER)

There is a subtle, largely unavoidable leakage vector in the HNM pipeline that must be
acknowledged in the limitations section. The Phase 2 best checkpoint (selected by validation
loss) is used to mine hard negatives from the training set. The HNM retraining then uses
the SAME validation set for early stopping and checkpoint selection. This means the
validation set performs double duty: it selected the model that identified the hard
negatives, and it then selects the retrained model. The validation set has influenced the
training data composition (via the mining model it selected), introducing a weak dependency
between the validation partition and the HNM training data.

This dependency could lead to slightly optimistic validation metrics for the retrained
model. With only 563 validation images, a three-way validation split (one partition for
baseline model selection, one for HNM model selection, one held out) is impractical.

**Mitigation:** Report test-set results as the definitive evaluation, and acknowledge
this dependency in the limitations section. The multi-seed runs (Step 7) provide partial
mitigation by showing whether the effect is consistent across different random
initialisations.

**Verification cell (include in the notebook):**
```python
import os

train_non_flood = set(os.listdir(TRAIN_NON_FLOOD_DIR))
val_flood      = set(os.listdir(VAL_FLOOD_DIR))
val_non_flood  = set(os.listdir(VAL_NON_FLOOD_DIR))
test_flood     = set(os.listdir(TEST_FLOOD_DIR))
test_non_flood = set(os.listdir(TEST_NON_FLOOD_DIR))

held_out = val_flood | val_non_flood | test_flood | test_non_flood

leaked = train_non_flood & held_out
assert len(leaked) == 0, f"LEAKAGE DETECTED: {len(leaked)} files overlap"

# Also verify augmented HNM images are not in val/test
hnm_dir = os.path.join(HNM_OUTPUT_PATH, "augmented_hard_negatives")
if os.path.exists(hnm_dir):
    hnm_files = set(os.listdir(hnm_dir))
    leaked_hnm = hnm_files & held_out
    assert len(leaked_hnm) == 0, f"HNM LEAKAGE: {len(leaked_hnm)} augmented files in held-out set"

print("Partition integrity verified: zero leakage detected.")
```

#### Pipeline

**Threshold notation (clarified):** The model outputs flood probability p_flood for each
image. A non-flood image is a hard negative if it receives a high flood probability:
p_flood > 0.7. Equivalently, its non-flood confidence is below tau = 0.3 (since
non-flood confidence = 1 - p_flood). The paper uses "tau = 0.3" as shorthand.
In code, the condition is simply `if predicted_flood_prob > 0.7`.

**IMPORTANT -- Paper Figure 2 diagram error:** The CIBB paper's Figure 2 flow chart
labels the HNM mining step as "Predict on test-set pools". This is a LABELING ERROR
in the diagram. The actual code and methodology text both mine from training data only.
The diagram must be corrected before journal submission. This is exactly the kind of
inconsistency that triggers reviewer suspicion of data leakage.

1. Load the trained baseline EfficientNetB0 (Phase 2 best checkpoint from Step 3).
2. **Mine (training set only):** Predict on every image in
   `processed_data/binary/train/non_flood/`. Flag any with p_flood > 0.7 as hard
   negatives. Log the filenames and their predicted probabilities to
   `results/logs/hnm_candidates.csv`.
3. **Augment:** Apply 5x augmentation to each hard negative using the same transform
   pipeline as training (rotation, flip, zoom, brightness). Save augmented copies to
   a dedicated directory: `processed_data/binary_hnm/augmented_hard_negatives/`.
4. **Inject:** Copy the full original training set into `processed_data/binary_hnm/` and
   add the augmented hard negatives to the `non_flood/` subdirectory. Val and test
   directories are NOT copied or modified -- the retraining notebook loads val and test
   from the original `processed_data/binary/{val,test}/` paths.
5. **Retrain Phase 1:** Last 30 layers, LR = 5e-5, up to 15 epochs. Validation monitored
   on the original, unmodified validation set.
6. **Retrain Phase 2:** Freeze first 50 layers only, LR = 1e-5, up to 10 epochs
   (epochs 16-25). Same unmodified validation set.
7. **Run the leakage verification cell** before evaluation.
8. Evaluate on the original, unmodified held-out test set (564 images).

#### Methodological Weakness: Single-Model Confirmation Bias

The current HNM pipeline mines hard negatives using a single trained checkpoint. This
means the model identifies negatives that are hard FOR ITSELF, which can reinforce its
own representational biases rather than targeting genuine semantic confusion. This is a
known limitation of offline HNM (Shrivastava et al. 2016).

**Mitigation (implement in Step 7 multi-seed runs):** When running multi-seed
experiments, perform CROSS-SEED mining:

1. Train baseline EfficientNetB0 with seed A.
2. Mine hard negatives using the model from seed B (a different seed's checkpoint).
3. Retrain seed A's model with the hard negatives identified by seed B.

This decorrelates the mining signal from the training signal and produces more robust
hard negative identification. Report results for both self-mining (current approach) and
cross-seed mining.

Additionally, report the UNION of hard negatives found across all 5 seeds. If different
seeds identify different hard negatives, that indicates the mining process is sensitive to
initialisation, which weakens the claim of a principled methodology. If all seeds converge
on the same hard negatives, that strengthens the claim.

#### Why 2 Hard Negatives Produce a 13% Accuracy Jump (Honest Assessment)

The original results show baseline EfficientNetB0 at 85.46% jumping to 98.76% after
HNM with only 2 mined images. This is suspicious and reviewers will flag it. There
are four contributing factors that must be discussed honestly in the paper:

1. **The preprocessing bug.** The baseline EfficientNetB0 (85.46%) was trained with
   double-rescaling, which did not merely compress the input distribution -- it rendered
   transfer learning non-functional. The frozen backbone layers received near-zero
   activations and contributed almost nothing to feature extraction. The 85.46% baseline
   is really the performance of a partially random network, not a fine-tuned model.
   The HNM model was ALSO trained with the bug, but it received a full retraining cycle
   with different hyperparameters (lower LR, more epochs). Much of the "HNM improvement"
   is almost certainly the benefit of a second training pass with better optimisation,
   not the hard negatives themselves. After fixing preprocessing, the baseline will likely
   be substantially higher, and the HNM delta will shrink significantly.

2. **The retraining effect (CRITICAL CONFOUND).** HNM retraining is effectively Phase 3 +
   Phase 4 of training. The baseline trains for up to 35 epochs (15 Phase 1 + 20
   Phase 2). The HNM model then trains for an additional 25 epochs (15 HNM Phase 1 +
   10 HNM Phase 2), for a total of up to 60 epochs -- nearly double the training budget.
   The 2 hard negative images (producing 10 augmented copies in a dataset of 2,640)
   represent 0.38% of the training data. It is not plausible that 10 images drive a
   13-point accuracy gain. An honest comparison requires an ablation where the baseline
   receives the same number of total training epochs WITHOUT HNM injection (Step 8,
   "Extended training without HNM"). **This ablation is the single most important
   experiment in the pipeline.** If it closes most of the gap, the core contribution
   must be reframed.

3. **Small dataset variance.** On 564 test images, a 13% accuracy difference is ~73
   images. With bootstrapping, the confidence intervals on these numbers may overlap
   more than expected. The multi-seed runs (Step 7) will clarify this.

4. **The architectural comparison confound.** The CIBB paper claims that "EfficientNetB0's
   compound scaling approach proves more compatible with hard negative mining than
   ResNet50's residual hierarchy." This claim is unsupported because ResNet50 was never
   trained with HNM. The comparison is between an architecture with additional training
   and hard negative injection versus one without either. To support an architectural
   interaction claim, ResNet50+HNM results are required, and the HNM improvement DELTA
   must be shown to differ between the two architectures. Without this, remove the
   claim from the paper entirely.

The paper should discuss all four factors in a "Limitations" section.

### Step 6 -- Evaluation and Comparison

**Notebook:** `06_evaluation.ipynb`
**Compute:** Colab T4 for inference; CPU for metrics and plots.

For every model (Baseline EfficientNetB0, Baseline ResNet50, EfficientNetB0+HNM,
**and ResNet50+HNM**):
- Accuracy, Precision, Recall, F1, AUC-ROC, **PR-AUC** (precision-recall AUC).
- **PR-AUC is the primary discrimination metric** for the paper, not ROC-AUC.
  ROC-AUC can be misleadingly high with class imbalance in disaster detection.
  PR-AUC directly measures the trade-off that matters operationally.
- Confusion matrix (with cell-count annotation).
- Per-class classification report.
- ROC and Precision-Recall curves.
- Save predictions CSV: `filename, true_label, predicted_label, probability, correct,
  is_swimming_pool, flood_severity`.

**ResNet50+HNM (NEW -- REQUIRED):** Apply the same HNM pipeline from Step 5 to the
baseline ResNet50 model. This experiment is necessary to resolve the confounded
architectural comparison. Specifically:

1. Use the ResNet50 Phase 2 best checkpoint to mine hard negatives from
   `train/non_flood/` with the same tau = 0.3 threshold.
2. Apply the same 5x augmentation and injection protocol.
3. Retrain with the same HNM schedule (Phase 1: LR = 5e-5, 15 epochs; Phase 2: LR =
   1e-5, 10 epochs).
4. Evaluate on the unmodified test set.

Report the HNM improvement delta for both architectures:
- Delta_EfficientNetB0 = (EfficientNetB0+HNM accuracy) - (EfficientNetB0 baseline accuracy)
- Delta_ResNet50 = (ResNet50+HNM accuracy) - (ResNet50 baseline accuracy)

If the deltas are similar, the HNM benefit is architecture-agnostic and the architectural
interaction claim should be removed. If they differ substantially, the claim is supported
but must be contextualised with the caveat that two architectures cannot support general
claims about architectural families.

**Swimming pool false positive rate WITH binomial confidence intervals AND significance
test:**
With only 15 pool images in the test set, a 0% FP rate has a 95% Clopper-Pearson
confidence interval of approximately [0%, 21.8%]. This must be reported honestly.
Do not claim "complete elimination" without stating the CI.

Additionally, compute the one-sided Fisher exact test p-value comparing the baseline
pool FP rate (e.g., 3/15 for EfficientNetB0) against the HNM pool FP rate (0/15). At
3/15 vs 0/15, p is approximately 0.11 (not significant at alpha = 0.05). The swimming
pool result is suggestive evidence, not a confirmed finding at conventional significance
levels.

```python
from scipy.stats import binom, fisher_exact

def binomial_ci(successes: int, trials: int, alpha: float = 0.05):
    """Clopper-Pearson exact binomial confidence interval."""
    from scipy.stats import beta
    lo = beta.ppf(alpha / 2, successes, trials - successes + 1) if successes > 0 else 0.0
    hi = beta.ppf(1 - alpha / 2, successes + 1, trials - successes) if successes < trials else 1.0
    return lo, hi

# Example: 0 FP out of 15 pools
lo, hi = binomial_ci(0, 15)
print(f"Pool FP rate: 0/15 = 0.0%, 95% CI: [{lo*100:.1f}%, {hi*100:.1f}%]")

# Fisher exact test: baseline 3/15 FP vs HNM 0/15 FP
# Contingency table: [[correct_baseline, fp_baseline], [correct_hnm, fp_hnm]]
table = [[12, 3], [15, 0]]
odds_ratio, p_value = fisher_exact(table, alternative='greater')
print(f"Fisher exact test (one-sided): p = {p_value:.4f}")
print(f"  {'Significant' if p_value < 0.05 else 'NOT significant'} at alpha = 0.05")
```

**Severity-stratified evaluation:** Since the dataset preserves flood severity metadata
(MajorFlood, ModerateFlood, MinorFlood), report recall broken down by severity. This
adds geoscience value because minor floods are harder to detect and more operationally
ambiguous. If the model's false negatives cluster in MinorFlood, that is a meaningful
finding for emergency response triage design.

**Extended training ablation (controls for the performance jump):** To distinguish
the effect of HNM injection from the effect of additional training epochs, add one
control run: retrain the baseline model for the same total number of epochs as the
HNM model (Phase 1 + Phase 2 + HNM Phase 1 + HNM Phase 2) but WITHOUT injecting
hard negatives. If this control performs similarly to HNM, the jump is attributable to
extra training rather than HNM itself.

**Threshold optimisation correction (CIBB paper flaw):** The CIBB paper reports an
optimal threshold of 0.85, selected by sweeping thresholds on the TEST set. This
constitutes test-set tuning and must not be repeated. In the journal submission, all
threshold selection must occur on the validation set per Step 13.

---

## Additional Experiments Required for Journal Submission

The baseline study (Steps 1-6) was originally a CIBB 2026 conference paper. The following
additional experiments are REQUIRED to meet Computers & Geosciences standards.

### Step 7 -- Statistical Robustness (Multi-Seed Runs)

**Notebook:** `07_multi_seed_runs.ipynb`
**Compute:** Colab T4 GPU (budget ~3-4 sessions across multiple days).

- Rerun the full EfficientNetB0+HNM pipeline with 5 different random seeds
  (42, 123, 256, 512, 1024).
- Report mean +/- standard deviation for all metrics.
- Apply McNemar's test or bootstrap confidence intervals to compare model pairs.
- This is essential: single-run results on 564 test images are not statistically robust.

### Step 8 -- Ablation Studies

**Notebook:** `08_ablation_studies.ipynb`
**Compute:** Colab T4 GPU.

Run these ablations and record results in a structured table:

| Ablation                          | What to vary                                      | Priority |
|-----------------------------------|---------------------------------------------------|----------|
| **Extended training without HNM** | **Retrain baseline for the same total epochs as HNM model (Phase1 + Phase2 + HNM-Phase1 + HNM-Phase2) WITHOUT injecting hard negatives. This is the CRITICAL control that isolates HNM's contribution from the benefit of additional training time.** | **RUN FIRST** |
| Augmentation factor               | 2x, 5x (baseline), 10x hard negative augmentation | High |
| HNM confidence threshold          | tau = 0.2, 0.3 (baseline), 0.4, 0.5 -- select best on VALIDATION set, then report that single configuration on test | High |
| Phase contribution                | Phase 1 only, Phase 2 only, both (baseline)        | Medium |
| Progressive vs standard fine-tune | Compare progressive unfreezing against unfreezing all layers from start | Medium |
| With vs without class weights     | Remove class weighting and compare                 | Medium |
| Head complexity                   | GAP->Dense(1) vs GAP->Dense(128)->Dense(1) vs GAP->Dense(256)->Dense(1) (baseline). The Dense(256) head adds ~327k parameters; on 2,627 training images this may overfit. | Medium |
| HNM augmentation diversity        | Compare: (a) same-transform 5x (baseline), (b) mixup between hard negatives and random non-flood images, (c) copy-paste of pool water region onto non-flood backgrounds. | Low |

### Step 9 -- Additional Baseline Comparisons

**Notebook:** `09_additional_baselines.ipynb`
**Compute:** Colab T4 GPU.

Add at least two more backbone comparisons using the same classification head and
training protocol:

1. **MobileNetV3-Small or MobileNetV3-Large** -- relevant because the paper frames
   deployment for emergency response on resource-constrained devices.
2. **A Vision Transformer (ViT-B/16 or Swin-T via `timm` or `keras_cv`)** -- reviewers
   will expect a transformer baseline given the current state of the field.

Also compare against at least one alternative hard negative handling strategy:
- **Focal Loss** (gamma=2.0) as a drop-in replacement for binary cross-entropy.
- Or **OHEM (Online Hard Example Mining)** during training.

### Step 10 -- Error Analysis and Interpretability

**Notebook:** `10_error_analysis.ipynb`
**Compute:** Colab T4 GPU for Grad-CAM; CPU for analysis.

1. **Grad-CAM visualisations:** For all three original models (and the best new baseline),
   generate heatmaps for:
   - Correctly classified flood images (one per severity level).
   - Correctly classified swimming pool images.
   - All misclassified images from the EfficientNetB0+HNM model.
2. **Failure case analysis:** For each misclassification, describe the image content, identify
   likely causes, and discuss what visual features confused the model.
3. **Feature space visualisation:** Use t-SNE or UMAP on the penultimate layer (Dense 256)
   embeddings for the full test set, coloured by true label and with swimming pools
   highlighted distinctly.

### Step 11 -- Cross-Dataset Generalisation

**Notebook:** `11_cross_dataset_validation.ipynb`
**Compute:** Colab T4 GPU.

**Domain matching is critical.** Your model is trained on street-level imagery. External
datasets must also be street-level or ground-level. Testing on aerial/satellite imagery
(FloodNet, MediaEval Satellite Task) will produce extreme performance drops that do not
reflect genuine generalisation failure -- they reflect a domain shift that no street-level
model should be expected to handle. Reviewers will dismiss this as not meaningful.

**Priority external datasets (street-level / ground-level):**

1. **Crisis Vision Benchmark (CrisisMMD / CrisisBench):** Social media images from
   multiple disaster events including floods. Ground-level, crowd-sourced, diverse
   geographic origins. This is the best match for your training domain.
2. **ASONAM / Damage Identification datasets:** Social media flood images from Twitter
   and Flickr, classified by damage severity.
3. **European Flood 2013 social media images** (if the ground-level subset is available).

**Secondary external datasets (acknowledge domain mismatch explicitly):**

4. **FloodNet** (University of Maryland): aerial/drone imagery. Include as a domain-shift
   stress test, but frame it clearly: "We additionally evaluate on aerial imagery to
   characterise the expected performance degradation under domain shift, not to claim
   cross-domain generalisation."

**Evaluation protocol:**
- Report performance WITHOUT retraining (zero-shot transfer).
- Report performance with lightweight adaptation (freeze backbone, retrain head only
  on a small sample from the external dataset) to measure adaptation efficiency.
- For each external dataset, report which failure modes dominate (false negatives on
  which flood severity? false positives on which scene types?).
- This addresses the critical reviewer concern about single-source generalisability.

**If no suitable external dataset can be obtained:** At minimum, perform a leave-one-
category-out analysis on the existing dataset. Train on all flood categories except one
(e.g., hold out MinorFlood entirely), test on the held-out category. This simulates
encountering a previously unseen flood presentation.

### Step 12 -- Computational Profiling

**Notebook:** `12_computational_profile.ipynb`
**Compute:** Colab T4 GPU for GPU metrics; CPU for CPU-only inference timing.

Report for each model:
- Parameter count and FLOPs (use `keras` model summary + a profiling library).
- Single-image inference time (mean of 100 runs, GPU and CPU separately).
- Batch inference throughput (images/second at batch_size=32).
- Peak GPU memory usage during training and inference.
- Model file size on disk (.keras format).

### Step 13 -- Threshold Optimisation (Expanded)

**Notebook:** `13_threshold_optimisation.ipynb`
**Compute:** CPU.

- Sweep thresholds from 0.05 to 0.95 in steps of 0.01 for each model.
- Plot threshold vs F1, threshold vs flood recall, threshold vs swimming pool FP rate.
- **Select the optimal threshold on the VALIDATION set.** Identify the operating point that
  maximises flood recall subject to 0% swimming pool false positives on validation data.
- Report the selected threshold's performance on the TEST set exactly once.
- Report the threshold range (on validation) within which the model maintains >= 95%
  flood recall.

**NOTE:** The CIBB paper's threshold of 0.85 was selected on the test set, which
constitutes test-set tuning. That approach must not be repeated. All threshold selection
in the journal submission uses validation data only.

### Step 14 -- Test-Time Augmentation (TTA)

**Notebook:** `14_test_time_augmentation.ipynb`
**Compute:** Colab T4 GPU.

TTA is standard practice in remote sensing and geoscience image classification. During
inference, each test image is evaluated multiple times under different transforms, and the
predictions are averaged. This reduces sensitivity to viewpoint, lighting, and orientation
artifacts that are common in street-level flood imagery captured by phones, dashcams, and
CCTV with varying angles.

**TTA transform set (applied independently, then averaged):**

| Transform ID | Description                                  |
|--------------|----------------------------------------------|
| original     | No modification (baseline prediction)        |
| hflip        | Horizontal flip                              |
| bright_up    | Brightness factor 1.15                       |
| bright_down  | Brightness factor 0.85                       |
| rotate_5     | Rotation +5 degrees, reflect fill            |

**Semantic validity note:** TTA is standard in satellite and aerial remote sensing where
rotation invariance is physically justified (satellite passes at varying angles). For
ground-level street imagery, some transforms have weaker justification: horizontal flip
is reasonable (streets are bilaterally symmetric), but rotation can produce unrealistic
camera perspectives. The rotation is limited to 5 degrees (not the 15 degrees used in
training augmentation) precisely because larger rotations are implausible at test time.
Brightness adjustments are well-justified because street-level imagery varies widely in
lighting conditions. If a reviewer objects to rotation in TTA, the ablation (below)
quantifies its marginal contribution.

**Implementation:**

IMPORTANT: TTA transforms must operate on images in the same pixel range that the
model expects. If using Option A (include_preprocessing=True), feed raw [0, 255]
pixels into TTA transforms and the model. If using Option B, feed [0, 1] pixels.
The brightness transforms below assume [0, 1] range -- adjust clip bounds if using
raw pixels.

```python
import numpy as np
from tensorflow.keras.preprocessing.image import (
    img_to_array, load_img, apply_affine_transform
)
from PIL import ImageEnhance

TTA_TRANSFORMS = {
    "original":    lambda img: img,
    "hflip":       lambda img: np.fliplr(img),
    "bright_up":   lambda img: np.clip(img * 1.15, 0.0, 1.0),
    "bright_down": lambda img: np.clip(img * 0.85, 0.0, 1.0),
    "rotate_5":    lambda img: apply_affine_transform(
                       img, theta=5, fill_mode="reflect"
                   ),
}

def predict_with_tta(
    model,
    image: np.ndarray,
    transforms: dict = TTA_TRANSFORMS,
) -> tuple[float, dict[str, float]]:
    """Return (mean_probability, per_transform_probabilities)."""
    preds = {}
    for name, fn in transforms.items():
        augmented = fn(image.copy())
        prob = float(model.predict(
            np.expand_dims(augmented, axis=0), verbose=0
        )[0, 0])
        preds[name] = prob
    mean_prob = float(np.mean(list(preds.values())))
    return mean_prob, preds
```

**What to report:**
- Full test-set accuracy, precision, recall, F1, AUC with and without TTA side by side.
- Swimming pool FP rate with and without TTA.
- Per-image comparison: scatter plot of single-pass probability vs TTA-averaged probability,
  coloured by correctness. Highlight cases where TTA flipped the prediction.
- TTA variance per image (std across the 5 transforms) as a measure of prediction stability.
  High-variance images are operationally unreliable and should be flagged for human review.
- Ablation: report results for subsets of transforms (e.g., original + hflip only) to show
  the marginal value of each augmentation.

**Interaction with threshold optimisation:** Re-run the threshold sweep from Step 13 using
TTA-averaged probabilities. The optimal threshold may shift when predictions are smoothed.

### Step 15 -- Probability Calibration

**Notebook:** `15_probability_calibration.ipynb`
**Compute:** CPU (no GPU needed; works on saved prediction arrays).

Flood detection is a decision support system. Emergency responders act on predicted
probabilities, not just binary labels. A model that outputs 0.90 flood probability should
be correct ~90% of the time at that confidence level. If calibration is poor, operators
either over-trust or under-trust the system, both of which cost lives or waste resources.

**Metrics to report:**

1. **Expected Calibration Error (ECE):** Partition predictions into B=10 equal-width bins
   by predicted probability (10 bins, not 15 -- with 563 validation images, 15 bins yields
   ~37 samples per bin on average, with some bins nearly empty; 10 bins gives ~56 per
   bin which is more stable). For each bin, compute the absolute difference between mean
   predicted probability and observed frequency of the positive class. ECE is the
   weighted average of these differences:
   ```
   ECE = sum_{b=1}^{B} (n_b / N) * |accuracy_b - confidence_b|
   ```
   Report ECE for each model, with and without TTA.
   **Report bootstrap 95% CIs on ECE** (resample predictions 1000 times, compute ECE
   for each resample, take 2.5th and 97.5th percentiles).

2. **Maximum Calibration Error (MCE):** The worst-case bin-level miscalibration.
   ```
   MCE = max_{b} |accuracy_b - confidence_b|
   ```

3. **Reliability diagram:** Plot observed frequency vs predicted probability for each bin.
   A perfectly calibrated model falls on the diagonal. Include histograms showing the
   number of predictions per bin beneath the diagram.

**Calibration methods to apply and compare:**

- **Platt scaling (sigmoid):** Fit a logistic regression on the validation set's raw logits
  (pre-sigmoid activations) to learn a temperature and bias. Apply the fitted scaler to
  test logits. This is the standard post-hoc calibration for binary classifiers.
  **IMPORTANT:** The model's final layer is `Dense(1, sigmoid)`. The model.predict()
  output is a probability, NOT a logit. To extract logits, build a secondary model that
  outputs the pre-sigmoid activation:
  ```python
  from tensorflow.keras.models import Model

  # Build a logit-extraction model
  logit_layer = model.layers[-1]  # final Dense(1, sigmoid)
  # Create a model that outputs the pre-activation value
  logit_model = Model(
      inputs=model.input,
      outputs=model.layers[-2].output  # output of Dropout(0.3) layer
  )
  # Then apply the Dense weights manually, or alternatively:
  # Rebuild the last layer without sigmoid
  import tensorflow as tf
  logit_output = tf.keras.layers.Dense(
      1, activation=None, name="logit_output"
  )(model.layers[-2].output)
  logit_model = Model(inputs=model.input, outputs=logit_output)
  logit_model.layers[-1].set_weights(model.layers[-1].get_weights())

  val_logits = logit_model.predict(val_generator).flatten()
  test_logits = logit_model.predict(test_generator).flatten()
  ```
  Fitting Platt scaling on probabilities instead of logits will produce incorrect
  calibration because the logistic regression expects unbounded inputs.
- **Temperature scaling:** A special case of Platt scaling where only a single temperature
  parameter T is learned. Divide logits by T before applying sigmoid. Fit T on
  validation logits by minimising NLL.
- **Isotonic regression:** Non-parametric calibration. Fit on validation set PROBABILITIES
  (not logits) and true labels using `sklearn.isotonic.IsotonicRegression`. This is the
  one method that correctly operates on probability outputs directly.

**Implementation sketch:**
```python
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import numpy as np

def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, list[dict]]:
    """Compute ECE and per-bin statistics."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob > bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        bin_size = mask.sum()
        gap = abs(bin_acc - bin_conf)
        ece += (bin_size / len(y_true)) * gap
        bins.append({
            "bin_lower": bin_edges[i],
            "bin_upper": bin_edges[i + 1],
            "accuracy": bin_acc,
            "confidence": bin_conf,
            "count": int(bin_size),
            "gap": gap,
        })
    return ece, bins

def platt_scaling(
    val_logits: np.ndarray,
    val_labels: np.ndarray,
    test_logits: np.ndarray,
) -> np.ndarray:
    """Fit Platt scaling on validation logits, apply to test logits."""
    lr = LogisticRegression(C=1e10, solver="lbfgs")
    lr.fit(val_logits.reshape(-1, 1), val_labels)
    return lr.predict_proba(test_logits.reshape(-1, 1))[:, 1]
```

**Critical data handling:**
- Platt scaling and isotonic regression are fit on the VALIDATION set only. They are
  then applied to the test set. Fitting on the test set is data leakage.
- If the validation set is small (563 images), report whether isotonic regression
  overfits by comparing val-ECE vs test-ECE. Platt scaling is typically more stable
  with limited calibration data.
- Save calibrated probabilities alongside raw probabilities in the predictions CSV so
  both can be audited.

**What to report in the paper:**
- ECE and MCE table: raw model, after Platt scaling, after temperature scaling,
  after isotonic regression -- for each of the three main models, with and without TTA.
- Reliability diagrams (one per model, pre- and post-calibration).
- A brief discussion of why calibration matters for operational flood response: threshold
  selection, confidence-based triage, and trust in automated alerts.

### Step 16 -- Flood Severity Stratified Evaluation

**Notebook:** `16_severity_evaluation.ipynb`
**Compute:** CPU (works on saved prediction CSVs).

The binary collapse (flood vs non-flood) discards flood severity information that is
central to the geoscience contribution. Reviewers at Computers & Geosciences will ask
why severity is ignored when it is available in the data. This step recovers that
information without requiring a new model.

**Using the existing binary model:**
The test set images retain their original category names (MajorFlood, ModerateFlood,
MinorFlood) in the filename prefixes and in the split metadata CSV. Use these to
stratify the binary predictions by severity.

Report for each model:
- Flood recall broken down by severity: recall_major, recall_moderate, recall_minor.
- False negative analysis by severity: which severity level has the most missed floods?
  If MinorFlood has lower recall, discuss the operational implication -- minor floods
  are visually ambiguous and may require different thresholds or human review.
- A table showing the distribution of false negatives across severity categories.
- Discuss whether the model's errors correlate with geoscience-relevant factors (flood
  depth, water extent, urban vs suburban context).

**Optional multi-class extension (if time permits):**
Train a 4-class model (MajorFlood, ModerateFlood, MinorFlood, NonFlood) using the same
backbone and head architecture. Compare against the binary model's severity-stratified
recall. Even negative results are useful: if multi-class accuracy is substantially lower,
this justifies the binary approach for first-stage triage.

### Step 17 -- Domain Robustness Analysis

**Notebook:** `17_domain_robustness.ipynb`
**Compute:** Colab T4 GPU.

Test the model's robustness to conditions it may encounter in operational deployment but
that may be underrepresented in the training data.

**Synthetic domain shifts (applied to the test set at inference time):**

| Domain shift         | Implementation                                     | Operational relevance |
|----------------------|----------------------------------------------------|-----------------------|
| Low light / dusk     | Reduce brightness to 0.3-0.5x                     | Floods often occur during storms with poor lighting |
| Heavy rain overlay   | Add synthetic rain streaks via image processing    | Camera lens during active precipitation |
| Colour cast (muddy)  | Shift hue toward brown/yellow                      | Muddy floodwater vs clear pool water |
| JPEG compression     | Recompress at quality=20                           | Social media and CCTV compression artifacts |
| Resolution reduction | Downscale to 112x112 then upscale back to 224x224 | Low-resolution surveillance cameras |

For each shift, report accuracy, recall, and pool FP rate. Plot a degradation curve
showing how each metric changes as the shift intensity increases.

**Cross-event analysis (if image metadata permits):** If the dataset's flood images
come from identifiable events (e.g., specific hurricanes, regional floods), perform a
leave-one-event-out evaluation. Train on all events except one, test on the held-out
event. This directly measures geographic and temporal generalisation and is the single
most convincing experiment for geoscience reviewers.

---

## Known Limitations (Discuss Honestly in the Paper)

Reviewers respect transparent discussion of limitations far more than they respect
inflated claims. The paper MUST include a limitations section covering these points.

1. **Dataset size and source diversity.** 3,754 images from a single institutional source
   (University of South Florida) is small for deep learning. The dataset's geographic,
   temporal, and photographer diversity is not documented. The model may have learned
   dataset-specific artifacts (camera angles, lighting conditions, location cues) rather
   than general flood features. Acknowledge this and position cross-dataset validation
   (Step 11) and domain robustness testing (Step 17) as partial mitigations.

2. **Random splitting vs event-based splitting.** The train/val/test split is random, not
   stratified by flood event or geographic location. If images from the same flood event
   appear in both train and test, the model may recognise the scene rather than the
   flood. This is a form of temporal leakage. Without event metadata, it cannot be
   fully mitigated. Acknowledge this as a limitation and recommend event-based splitting
   in future work with appropriately annotated datasets.

3. **Swimming pool sample size.** 15 pool images in the test set cannot support a
   statistically meaningful false positive rate. The 0% claim must always be accompanied
   by the binomial confidence interval ([0%, ~22%] at 95%) and the Fisher exact test
   p-value (approximately 0.11 for 3/15 vs 0/15, not significant at alpha = 0.05). The
   result is suggestive, not conclusive. Do not use language like "complete elimination"
   in the paper.

4. **Hard negative mining scope.** Only 2 hard negatives were mined, both swimming pools.
   The method has not been tested against other semantically confusing categories (wet
   roads, lakes, rain-soaked surfaces, irrigated fields, coastal scenes). The claim of a
   "generalizable HNM pipeline" is not yet supported by the evidence.

5. **Performance jump attribution.** The baseline-to-HNM improvement must be decomposed
   into: (a) preprocessing bug fix effect, (b) additional training epochs effect, (c)
   actual HNM contribution. The extended-training-without-HNM ablation (Step 8) is the
   critical experiment that isolates the HNM contribution. Without it, the improvement
   cannot be attributed to HNM.

6. **Geoscience reasoning.** The model performs image classification, not flood mapping,
   depth estimation, or hydrological analysis. Frame it as a first-stage filter within
   a larger geoscience pipeline, not as a standalone flood detection system.

7. **Architectural comparison confound.** The CIBB paper's claim that EfficientNetB0 is
   "more compatible" with HNM than ResNet50 is unsupported because HNM was only applied
   to EfficientNetB0. This is a comparison between one model with additional training
   and hard negative injection versus another model without either. ResNet50+HNM results
   are required to support any architectural interaction claim.

8. **Validation set double-duty in HNM pipeline.** The same validation set selects the
   baseline model (whose checkpoint drives HNM mining) and the retrained HNM model. This
   introduces a weak dependency between the validation partition and the HNM training
   data, potentially producing slightly optimistic validation metrics. Test-set results
   are the definitive evaluation.

9. **Stratification gap in non-flood categories.** The stratification key covers flood
   severity and swimming pool status but does not individually stratify the 12 junk
   categories. Uneven distribution of visually diverse non-flood categories (animals,
   vehicles, buildings) across splits could affect evaluation. Per-category distributions
   across splits should be verified and reported.

10. **Geoscience framing.** For Computers & Geosciences, the paper must answer a
    geoscience question, not merely apply ML to geoscience-adjacent imagery. The current
    framing is primarily a computer vision contribution. Strengthen the connection to
    operational flood monitoring infrastructure (gauges, GIS, sensor networks), discuss
    the relationship between image-based detection and hydrological variables (depth,
    extent), and use geoscience terminology (inundation detection, pluvial/fluvial
    flooding, urban flood mapping, disaster response informatics). The severity-stratified
    evaluation (Step 16) helps, but must be connected to operational decision thresholds
    for emergency response.

---

## Data Leakage Prevention (Applies to Entire Pipeline)

Data leakage is the single most common reason reviewers reject applied ML papers. This
project has several leakage vectors that must be explicitly guarded against at every step.

### Partition Boundaries

The three data partitions (train / val / test) are created ONCE in Step 2 and are
immutable for the rest of the study.

```
train (2,627 images)  -- used for: model training, HNM mining, HNM augmentation injection
val   (563 images)    -- used for: early stopping, LR scheduling, threshold/tau selection,
                         calibration fitting (Platt/isotonic), hyperparameter selection
test  (564 images)    -- used for: FINAL evaluation ONLY, reported in paper tables
```

### Leakage Vectors and Their Mitigations

| Leakage vector | Risk | Mitigation |
|----------------|------|------------|
| HNM mining on val/test images | High. Mining identifies "hard" images; if val/test images are mined, the model has effectively seen them during retraining. | Mine ONLY from `train/non_flood/`. Assert zero filename overlap with val/test before retraining. |
| HNM augmented images leaking into val/test loaders | High. If augmented copies are placed in the val or test directory trees, evaluation is invalid. | Augmented HNM images go into a dedicated `binary_hnm/` directory. Val/test loaders always point to the original `binary/{val,test}/`. |
| Choosing tau by looking at test-set pool FP rate | Medium. The confidence threshold should be validated on val, not test. | Step 8 ablation sweeps tau using validation metrics. The test set is evaluated ONCE per configuration. |
| Calibration fitted on test data | Medium. Platt scaling or isotonic regression must be fitted on val. | Step 15 fits calibrators on val predictions, applies to test. |
| TTA transforms tuned on test results | Low-medium. If you keep adding transforms until test metrics improve, that is implicit test-set tuning. | Define the TTA transform set a priori (Step 14) based on domain knowledge of street-level imagery, not by optimising test accuracy. |
| Threshold tuned on test set | Medium. Reporting "optimal threshold = X" where X was chosen to maximise test F1 is circular. The CIBB paper's threshold of 0.85 was selected on the test set -- this must not be repeated. | Step 13 selects the threshold on the validation set. Test performance at that threshold is then reported as-is. |
| Multi-seed run selection bias | Low. If you only report the best seed out of 5, that is cherry-picking. | Step 7 reports mean +/- std across ALL seeds. Individual seed results go in an appendix. |
| Double-rescaling preprocessing | High. ImageDataGenerator `rescale=1./255` combined with backbone `include_preprocessing=True` compresses input ~255x below expected range. Model adapts during training but pretrained features are non-functional and results are fragile. | Use Option A or B from Step 3. Verify generator output range matches backbone expectations in every training notebook. |
| Repeated hyperparameter selection on single val set | Medium. Sweeping tau, augmentation factor, head size, phase schedule, class weights, and TTA transforms all on the same 563-image validation set makes it semi-training data. | Acknowledge this in the limitations section. With only 3,754 total images, a nested cross-validation or dedicated tuning split is impractical. Report validation-set performance alongside test-set performance for all configurations so readers can judge overfitting to val. |
| Validation set double-duty in HNM pipeline | Medium. The validation set selects both the baseline checkpoint (which drives HNM mining) and the retrained HNM checkpoint. This introduces a weak dependency between the validation partition and the HNM training data. | Acknowledge in limitations. Use test-set results as definitive evaluation. Multi-seed runs (Step 7) provide partial mitigation. |

### Verification Procedure

Every notebook that touches model predictions or data generators must include a
verification cell at the top that:

1. Loads the file lists for train, val, and test from the original split manifest.
2. Asserts pairwise zero intersection among the three sets.
3. If HNM augmentation has been applied, asserts the augmented files exist only within
   the training tree.
4. Prints partition sizes and a confirmation message.

Failure of any assertion halts the notebook. Do not bypass with try/except.

---

## Code Standards

### File Organisation

```
imagevalidation/
  notebooks/
    01_data_exploration.ipynb
    02_stratified_splitting.ipynb
    03_baseline_efficientnetb0.ipynb
    04_baseline_resnet50.ipynb
    05_hard_negative_mining.ipynb
    06_evaluation.ipynb
    07_multi_seed_runs.ipynb
    08_ablation_studies.ipynb
    09_additional_baselines.ipynb
    10_error_analysis.ipynb
    11_cross_dataset_validation.ipynb
    12_computational_profile.ipynb
    13_threshold_optimisation.ipynb
    14_test_time_augmentation.ipynb
    15_probability_calibration.ipynb
    16_severity_evaluation.ipynb
    17_domain_robustness.ipynb
  scripts/
    train_efficientnet.py
    train_resnet50.py
    hard_negative_mining.py
    evaluate.py
    grad_cam.py
    tta.py
    calibration.py
    utils.py
  configs/
    efficientnet_config.yaml
    resnet50_config.yaml
    hnm_config.yaml
  results/
    figures/
    tables/
    logs/
    predictions/
  data/
    README.md            (download instructions, NOT raw data)
  docker/
    Dockerfile           (pin CUDA, TF, Python versions for reproducibility)
    environment.yml      (conda environment for local CPU work)
  requirements.txt
  LICENSE                (MIT or Apache-2.0)
  README.md
  claude.md              (this file)
```

### Naming Conventions

- Notebooks: `{NN}_{descriptive_slug}.ipynb` (e.g., `07_multi_seed_runs.ipynb`).
- Scripts: `snake_case.py`.
- Model checkpoints: `{architecture}_{phase}_{timestamp}.keras`.
- Result files: `{experiment}_{metric}_{timestamp}.csv`.
- Figures: `fig_{number}_{short_description}.png` (300 DPI minimum for journal).

### Code Style

- Python 3.10+.
- Use type hints on all function signatures.
- Every notebook begins with a markdown cell stating its purpose, required compute
  (GPU/CPU), estimated runtime, and dependencies.
- Every notebook begins with a code cell that verifies the runtime environment:
  ```python
  import subprocess
  gpu_info = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
  if "T4" in gpu_info.stdout:
      print("Runtime: Colab T4 GPU")
  elif "failed" in gpu_info.stderr.lower() or gpu_info.returncode != 0:
      print("Runtime: CPU only")
  else:
      print(f"Runtime: Local GPU detected\n{gpu_info.stdout[:200]}")
  ```
- Group imports: stdlib, then third-party, then project-local. No wildcard imports.
- Constants in UPPER_SNAKE_CASE at the top of each notebook/script.
- Use `os.path.join` or `pathlib.Path` for all file paths. Never hardcode absolute paths
  without a configurable base.
- Seed EVERYTHING: `np.random.seed`, `tf.random.set_seed`, `random.seed`, and
  `PYTHONHASHSEED` environment variable.
- Add docstrings to all functions.
- No dead code, no commented-out blocks left behind, no print-debugging in final versions.

### Reproducibility Checklist

- [ ] All random seeds are fixed and documented.
- [ ] `requirements.txt` pins exact versions (`tensorflow==2.16.1`, not `tensorflow>=2.0`).
- [ ] `docker/Dockerfile` provided that pins CUDA version, TF version, and Python version.
      Colab's environment changes without notice; a Dockerfile ensures results can be
      reproduced outside Colab. At minimum, record `!nvidia-smi`, `python --version`,
      `pip freeze` output at the start of every GPU notebook.
- [ ] Every result figure and table can be regenerated by running the corresponding notebook
      end-to-end.
- [ ] Model checkpoints for the final reported results are saved to Drive and referenced in
      the README.
- [ ] The dataset download instructions are clear and complete.
- [ ] The open-source license file is present at the repository root.
- [ ] Partition integrity verified: train/val/test sets have zero filename overlap.
- [ ] Per-category distribution across splits verified and reported.
- [ ] HNM mining was performed on training data only (assertion cell passes).
- [ ] HNM confidence threshold (tau) was selected using validation metrics, not test metrics.
- [ ] Calibration (Platt/isotonic) was fitted on validation predictions, not test predictions.
- [ ] Platt scaling uses pre-sigmoid logits, not post-sigmoid probabilities.
- [ ] TTA transform set was defined a priori, not tuned to maximise test accuracy.
- [ ] Multi-seed results report mean +/- std across all seeds, not cherry-picked best run.
- [ ] ECE and reliability diagrams are included for all reported models.
- [ ] Preprocessing verified: no double-rescaling between ImageDataGenerator and backbone
      built-in preprocessing (see Step 3 bug documentation).
- [ ] Swimming pool FP rate reported with binomial confidence intervals AND Fisher exact
      test p-value.
- [ ] PR-AUC reported as primary discrimination metric (not just ROC-AUC).
- [ ] Extended-training-without-HNM ablation included to isolate HNM contribution.
- [ ] ResNet50+HNM included to resolve confounded architectural comparison.
- [ ] Paper Figure 2 corrected: HNM mining step must say "training set", not "test-set".
- [ ] Threshold selection performed on validation set, not test set (CIBB paper used test set).

---

## Journal-Specific Requirements (Computers & Geosciences)

### Mandatory for Acceptance

1. **Open-source code.** All code must be in a public repository with a clearly stated
   open-source license. Manuscripts describing non-open-source code are desk rejected.
2. **Code availability section.** The paper must include a section with: repository URL,
   license, language/framework versions, and instructions to reproduce results.
3. **5,000-word limit** for research papers (excluding abstract, references, figure captions).
4. **Single-column, double-spaced** manuscript format.
5. **Author-date reference style** (not numbered).
6. **Cover letter** confirming the submission follows all requirements.
7. **Geoscience contribution.** The paper must answer a geoscience question, not merely
   apply ML to geoscience data. Frame the contribution around operational flood monitoring,
   emergency response pipeline integration, or hydrology. Reviewers will reject work that
   reads as pure computer vision.

### Scope Alignment -- How to Frame the Paper

- Position the system as a first-stage filter in a geoscience emergency response pipeline.
- Connect to real-world hydrology/flood monitoring workflows (e.g., integration with GIS,
  street-level sensor networks, or social media image triage).
- Discuss geographic and climatic generalisability as a geoscience concern, not just a
  model robustness concern.
- Use geoscience terminology: "inundation detection", "pluvial/fluvial flooding",
  "urban flood mapping", "disaster response informatics".
- Discuss what "MajorFlood" vs "MinorFlood" means in terms of observable flood
  characteristics and connect to emergency response decision thresholds.

### What Reviewers Will Scrutinise

Items marked [ADDRESSED] have corresponding experiments in the pipeline. Items marked
[LIMITATION] cannot be fully resolved but must be discussed honestly in the paper.

- [LIMITATION] Dataset size and diversity (3,754 images from one source is small).
  Mitigated by: cross-dataset validation (Step 11), domain robustness (Step 17).
- [ADDRESSED] Single-run results without confidence intervals. Fixed by: multi-seed
  runs (Step 7) with bootstrap CIs.
- [LIMITATION] Only 2 hard negatives mined. Mitigated by: cross-seed mining (Step 5),
  HNM augmentation diversity ablation (Step 8), honest limitations discussion.
- [ADDRESSED] Only 2 backbone architectures compared. Fixed by: MobileNetV3, ViT,
  focal loss baselines (Step 9).
- [ADDRESSED] No interpretability analysis. Fixed by: Grad-CAM, t-SNE/UMAP (Step 10).
- [ADDRESSED] No cross-dataset validation. Fixed by: Step 11 (with domain-appropriate
  datasets, not aerial imagery).
- [ADDRESSED] No computational cost analysis. Fixed by: Step 12.
- [LIMITATION] Thin literature review. Expand to 40-60 references for journal version.
- [ADDRESSED] No probability calibration. Fixed by: Step 15 with logit extraction,
  bootstrap CIs, reliability diagrams.
- [ADDRESSED] No test-time augmentation. Fixed by: Step 14 with semantic justification.
- [ADDRESSED] HNM data leakage risk. Fixed by: leakage verification cells, Figure 2
  diagram correction, partition integrity assertions.
- [ADDRESSED] Preprocessing double-rescaling bug. Fixed by: Step 3 rewrite with
  verification cell.
- [ADDRESSED] Suspicious performance jump (85% -> 99%). Fixed by: extended-training-
  without-HNM ablation (Step 8), preprocessing bug acknowledgement, honest discussion
  in Known Limitations section.
- [ADDRESSED] No severity-stratified evaluation. Fixed by: Step 16.
- [LIMITATION] Random splitting vs event-based splitting. Cannot be fully fixed without
  event metadata. Acknowledged in Known Limitations.
- [ADDRESSED] Swimming pool FP rate statistically meaningless on n=15. Fixed by:
  binomial CIs and Fisher exact test reported alongside point estimate (Step 6).
- [ADDRESSED] PR-AUC missing as primary metric. Fixed by: Step 6.
- [ADDRESSED] Confounded architectural comparison. Fixed by: ResNet50+HNM (Step 6).
- [ADDRESSED] Threshold tuned on test set in CIBB paper. Fixed by: validation-only
  threshold selection (Step 13).
- [LIMITATION] Validation set double-duty in HNM pipeline. Acknowledged in Known
  Limitations. Multi-seed runs provide partial mitigation.
- [LIMITATION] Stratification gap in non-flood categories. Verification added to Step 2.
  Acknowledged in Known Limitations.
- [LIMITATION] Geoscience framing is thin. Must be strengthened for journal scope.

---

## Key Hyperparameters Reference

| Parameter                     | Baseline EfficientNetB0 | EfficientNetB0 + HNM | Baseline ResNet50 | ResNet50 + HNM |
|-------------------------------|-------------------------|-----------------------|-------------------|----------------|
| Backbone                      | EfficientNetB0 (5.3M)  | EfficientNetB0 (5.3M) | ResNet50 (25.6M) | ResNet50 (25.6M) |
| Input size                    | 224 x 224               | 224 x 224             | 224 x 224         | 224 x 224       |
| Batch size                    | 32                      | 32                    | 32                | 32              |
| Phase 1 LR                   | 1e-4                    | 5e-5                  | 1e-4              | 5e-5            |
| Phase 2 LR                   | 1e-5                    | 1e-5                  | 1e-5              | 1e-5            |
| Phase 1 epochs (max)         | 15                      | 15                    | 15                | 15              |
| Phase 2 epochs (max)         | 20                      | 10                    | 20                | 10              |
| Phase 1 trainable layers     | Last 30                 | Last 30               | Last 30           | Last 30         |
| Phase 2 frozen layers        | First 50                | First 50              | First 50          | First 50        |
| Dropout (post-GAP)           | 0.2                     | 0.2                   | 0.2               | 0.2             |
| Dropout (post-BN)            | 0.3                     | 0.3                   | 0.3               | 0.3             |
| Dense hidden units            | 256                     | 256                   | 256               | 256             |
| HNM threshold (tau)          | --                      | 0.3                   | --                | 0.3             |
| HNM augmentation factor      | --                      | 5x                    | --                | 5x              |
| EarlyStopping patience       | 7                       | 7                     | 7                 | 7               |
| ReduceLROnPlateau patience   | 3                       | 3                     | 3                 | 3               |
| ReduceLROnPlateau factor     | 0.5                     | 0.5                   | 0.5               | 0.5             |
| ReduceLROnPlateau min_lr     | 1e-7                    | 1e-7                  | 1e-7              | 1e-7            |
| Class weights                | sklearn balanced        | sklearn balanced      | sklearn balanced  | sklearn balanced |
| Optimizer                    | Adam                    | Adam                  | Adam              | Adam            |
| Loss                         | binary_crossentropy     | binary_crossentropy   | binary_crossentropy | binary_crossentropy |
| Random seed                  | 42                      | 42                    | 42                | 42              |

---

## Existing Results (from CIBB 2026 Conference Paper)

**WARNING:** These results were produced with the double-rescaling preprocessing bug
(see Step 3). They should be treated as reference baselines only. All experiments must
be re-run with corrected preprocessing for the journal submission. Do not report these
numbers in the Computers & Geosciences paper. Do not include them even as a "before fix"
reference, as readers may misinterpret the gap between bugged and corrected results as an
effect size attributable to the methodology.

**Additional note on the CIBB threshold result:** The reported optimal threshold of 0.85
was selected by sweeping on the test set. This constitutes test-set tuning and is not a
valid result. The journal submission must select thresholds on the validation set only.

| Model                 | Accuracy | Precision | Recall | F1     | AUC    | Pool FP Rate |
|-----------------------|----------|-----------|--------|--------|--------|--------------|
| EfficientNetB0        | 85.46%   | 90.88%    | 82.22% | 86.33% | 0.9359 | 20.00%       |
| ResNet50              | 94.15%   | 94.90%    | 94.60% | 94.75% | 0.9815 | 13.33%       |
| EfficientNetB0 + HNM  | 98.76%   | 98.43%    | 99.37% | 98.89% | 0.9993 | 0.00%        |

---

## Dependencies

```
tensorflow>=2.16,<2.18
numpy>=1.26
pandas>=2.2
scikit-learn>=1.4
matplotlib>=3.8
seaborn>=0.13
Pillow>=10.0
tqdm>=4.66
pyyaml>=6.0
tf-keras-vis>=0.8    # for Grad-CAM
umap-learn>=0.5      # for feature space visualisation
scipy>=1.12          # for statistical tests
netcal>=1.3          # for calibration metrics and methods (optional, sklearn suffices for basics)
```

Install in Colab at the top of each GPU notebook:
```python
%pip install -q tf-keras-vis umap-learn netcal
```

---

## Session Workflow (Day-to-Day)

1. Open VS Code. Open the target `.ipynb` notebook.
2. Select kernel: Colab (T4 GPU) for training notebooks, local Python for CPU notebooks.
3. Run the Drive mount cell.
4. Run the environment check cell (nvidia-smi, TensorFlow version, Drive accessibility).
5. Execute the notebook cells in order.
6. Verify that checkpoints and logs are saved to Drive before ending the session.
7. Commit notebook outputs and result files to Git at the end of each work session.