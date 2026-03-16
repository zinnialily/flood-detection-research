"""
train_hnm.py -- Hard Negative Mining retraining script.

Loads a baseline Phase 2 checkpoint, mines hard negatives from flagged
confounder categories, and retrains with them injected into the training set.

Supports three modes:
  - percentile (default): mine top-N% by flood probability from candidate list
  - sweep: ablation over multiple tau thresholds, pick best on val metrics
  - --no_injection: extended-training control (same epoch budget, no HNM)

Compute: Colab T4 GPU (or any CUDA-capable GPU).
Estimated runtime: ~1-3 hours depending on mode and architecture.
Dependencies: tensorflow>=2.16, scikit-learn, numpy, pandas.

Usage:
    # Percentile mode (default)
    python scripts/train_hnm.py --arch efficientnet \
        --model_path models/efficientnet_phase2_best.keras

    # Sweep mode (tau ablation)
    python scripts/train_hnm.py --arch efficientnet \
        --model_path models/efficientnet_phase2_best.keras \
        --tau_mode sweep

    # Extended-training control (no HNM injection)
    python scripts/train_hnm.py --arch efficientnet \
        --model_path models/efficientnet_phase2_best.keras \
        --no_injection
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import average_precision_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    img_to_array,
    load_img,
    save_img,
)
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (  # noqa: E402
    PREPROCESS_FN,
    build_callbacks,
    freeze_for_phase1,
    freeze_for_phase2,
    set_all_seeds,
    verify_preprocessing,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BATCH_SIZE: int = 32
IMG_SIZE: Tuple[int, int] = (224, 224)

HNM_PHASE1_LR: float = 5e-5
HNM_PHASE1_EPOCHS: int = 15
HNM_PHASE2_LR: float = 1e-5
HNM_PHASE2_EPOCHS: int = 10
TOTAL_HNM_BUDGET: int = HNM_PHASE1_EPOCHS + HNM_PHASE2_EPOCHS  # 25

HNM_AUG_FACTOR: int = 5

SWEEP_TAUS: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7]
SWEEP_HEAD_EPOCHS: int = 5

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Hard Negative Mining retraining script.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--arch",
        required=True,
        choices=list(PREPROCESS_FN.keys()),
        help="Backbone architecture: 'efficientnet' or 'resnet50'.",
    )
    parser.add_argument(
        "--model_path",
        required=True,
        type=str,
        help="Path to baseline Phase 2 checkpoint (.keras file).",
    )
    parser.add_argument(
        "--data_dir",
        default="./data/FloodingDataset2",
        type=str,
        help="Dataset root directory.",
    )
    parser.add_argument(
        "--output_dir",
        default="./models",
        type=str,
        help="Directory where model checkpoints are saved.",
    )
    parser.add_argument(
        "--tau_mode",
        default="percentile",
        choices=["percentile", "sweep"],
        help="Mining mode: 'percentile' (top-N%%) or 'sweep' (tau ablation).",
    )
    parser.add_argument(
        "--top_pct",
        default=0.10,
        type=float,
        help="Top percentage to mine in percentile mode (0.0-1.0).",
    )
    parser.add_argument(
        "--no_injection",
        action="store_true",
        help="Skip HNM injection; train on original data for same epoch budget.",
    )
    parser.add_argument(
        "--phase_boundary",
        default="30,50",
        type=str,
        help="Comma-separated n_trainable,n_frozen for phase boundaries.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def parse_phase_boundary(value: str) -> Tuple[int, int]:
    """Parse a 'n_trainable,n_frozen' string into a tuple of ints.

    Args:
        value: Comma-separated string with exactly two positive integers.

    Returns:
        Tuple (n_trainable, n_frozen).

    Raises:
        argparse.ArgumentTypeError: If parsing fails.
    """
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"--phase_boundary must be 'n_trainable,n_frozen', got '{value}'"
        )
    try:
        n_trainable, n_frozen = int(parts[0].strip()), int(parts[1].strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Both values in --phase_boundary must be integers, got '{value}'"
        ) from exc
    if n_trainable <= 0 or n_frozen <= 0:
        raise argparse.ArgumentTypeError(
            f"Both phase_boundary values must be > 0, got {n_trainable}, {n_frozen}"
        )
    return n_trainable, n_frozen


# ---------------------------------------------------------------------------
# Runtime environment check
# ---------------------------------------------------------------------------


def print_runtime_env() -> None:
    """Print GPU info, TensorFlow version, and Python version."""
    print("=" * 60)
    print("Runtime environment")
    print("=" * 60)
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout[:600])
    else:
        print("nvidia-smi not available (CPU-only runtime or no driver).")
    print(f"TensorFlow version : {tf.__version__}")
    print(f"Python version     : {sys.version}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Partition safety
# ---------------------------------------------------------------------------


def _collect_basenames_recursive(directory: str) -> set:
    """Collect all image basenames recursively from a directory.

    Args:
        directory: Path to scan.

    Returns:
        Set of basename strings.
    """
    basenames = set()
    for root, _dirs, files in os.walk(directory):
        for f in files:
            if Path(f).suffix.lower() in VALID_IMAGE_EXTENSIONS:
                basenames.add(f)
    return basenames


def verify_partition_safety(
    mining_files: set,
    val_dir: str,
    test_dir: str,
    hnm_augmented_dir: str | None = None,
) -> None:
    """Assert zero leakage between mining candidates and held-out sets.

    Hard fails (no try/except) if any overlap is detected.

    Args:
        mining_files: Set of basenames from mining_candidates file.
        val_dir: Path to validation split directory.
        test_dir: Path to test split directory.
        hnm_augmented_dir: Optional path to augmented HNM directory from
            a prior run. Checked for overlap if it exists.
    """
    val_files = _collect_basenames_recursive(val_dir)
    test_files = _collect_basenames_recursive(test_dir)
    held_out = val_files | test_files

    leaked = mining_files & held_out
    assert len(leaked) == 0, (
        f"LEAKAGE: {len(leaked)} mining files overlap val/test. Halting.\n"
        f"Leaked files: {sorted(leaked)[:10]}"
    )

    # Check augmented files from prior runs.
    if hnm_augmented_dir is not None and os.path.isdir(hnm_augmented_dir):
        aug_files = _collect_basenames_recursive(hnm_augmented_dir)
        assert len(aug_files & held_out) == 0, (
            "Augmented HNM files found in val/test. Halting."
        )

    print("[OK] Partition integrity verified: zero leakage detected.")


# ---------------------------------------------------------------------------
# Mining inference
# ---------------------------------------------------------------------------


def mine_candidates(
    model: tf.keras.Model,
    candidate_paths: List[str],
    arch: str,
) -> List[Tuple[str, float]]:
    """Run mining inference on candidate images.

    Uses the same preprocessing as analyze_confounders.py: raw [0, 255]
    pixels through PREPROCESS_FN[arch]. The model output is P(non_flood)
    due to alphabetical class ordering (flood=0, non_flood=1), so
    flood_prob = 1.0 - pred.

    Args:
        model: Loaded baseline Keras model.
        candidate_paths: List of absolute image paths.
        arch: Architecture key for preprocessing.

    Returns:
        List of (path, flood_prob) tuples.
    """
    preprocess_fn = PREPROCESS_FN[arch]
    results: List[Tuple[str, float]] = []

    for path in tqdm(candidate_paths, desc="Mining inference", unit="img"):
        try:
            img = load_img(path, target_size=IMG_SIZE)
            x = img_to_array(img)  # [0, 255], NO /255
            x = preprocess_fn(x[np.newaxis, ...])
            pred = float(model.predict(x, verbose=0)[0][0])  # P(non_flood)
            flood_prob = 1.0 - pred
            results.append((path, flood_prob))
        except Exception as exc:
            print(f"[WARN] Skipping {path}: {exc}")

    return results


# ---------------------------------------------------------------------------
# Augmentation and injection
# ---------------------------------------------------------------------------


def augment_hard_negatives(
    hard_negative_paths: List[str],
    hnm_augmented_dir: str,
    arch: str,
    aug_factor: int = HNM_AUG_FACTOR,
) -> List[str]:
    """Generate augmented copies of hard negative images.

    Applies strong augmentation (rotation=30, zoom=0.3, brightness=[0.6,1.4],
    horizontal flip) to each hard negative, saving aug_factor copies.

    Args:
        hard_negative_paths: Absolute paths to original hard negative images.
        hnm_augmented_dir: Directory to save augmented copies.
        arch: Architecture key for preprocessing (applied to saved copies
            for consistency, though the generator will re-preprocess).
        aug_factor: Number of augmented copies per image.

    Returns:
        List of paths to all augmented images.
    """
    os.makedirs(hnm_augmented_dir, exist_ok=True)

    strong_aug = ImageDataGenerator(
        rotation_range=30,
        zoom_range=0.3,
        brightness_range=[0.6, 1.4],
        horizontal_flip=True,
    )

    augmented_paths: List[str] = []

    for img_path in tqdm(hard_negative_paths, desc="Augmenting hard negatives"):
        img = load_img(img_path, target_size=IMG_SIZE)
        x = img_to_array(img)  # [0, 255]
        x = x[np.newaxis, ...]  # (1, 224, 224, 3)

        basename = Path(img_path).stem
        ext = Path(img_path).suffix

        aug_iter = strong_aug.flow(x, batch_size=1)
        for i in range(aug_factor):
            aug_img = next(aug_iter)[0]  # (224, 224, 3)
            # Clip to valid pixel range before saving.
            aug_img = np.clip(aug_img, 0.0, 255.0)
            out_name = f"{basename}_hnm_aug_{i}{ext}"
            out_path = os.path.join(hnm_augmented_dir, out_name)
            save_img(out_path, aug_img)
            augmented_paths.append(out_path)

    print(f"[INFO] Saved {len(augmented_paths)} augmented hard negatives to {hnm_augmented_dir}")
    return augmented_paths


def build_hnm_training_dir(
    original_train_dir: str,
    hnm_train_dir: str,
    augmented_paths: List[str],
) -> None:
    """Copy original training set and inject augmented hard negatives.

    Creates hnm_train_dir/{flood,non_flood}/ with:
    - All original training images (copied from original_train_dir)
    - Augmented hard negatives added to non_flood/

    Args:
        original_train_dir: Path to processed_data/binary/train/.
        hnm_train_dir: Destination path for the HNM training set.
        augmented_paths: List of augmented image paths to inject.
    """
    # Clean up prior HNM training directory.
    if os.path.isdir(hnm_train_dir):
        shutil.rmtree(hnm_train_dir)

    # Copy entire original training set.
    print(f"[INFO] Copying original training set to {hnm_train_dir} ...")
    shutil.copytree(original_train_dir, hnm_train_dir)

    # Inject augmented hard negatives into non_flood/.
    non_flood_dir = os.path.join(hnm_train_dir, "non_flood")
    os.makedirs(non_flood_dir, exist_ok=True)

    for aug_path in augmented_paths:
        dest = os.path.join(non_flood_dir, os.path.basename(aug_path))
        shutil.copy2(aug_path, dest)

    # Count final composition.
    flood_count = len(os.listdir(os.path.join(hnm_train_dir, "flood")))
    non_flood_count = len(os.listdir(non_flood_dir))
    print(
        f"[INFO] HNM training set: flood={flood_count}, "
        f"non_flood={non_flood_count} (original + {len(augmented_paths)} augmented)"
    )


def cleanup_hnm_dir(hnm_dir: str) -> None:
    """Remove a temporary HNM directory tree.

    Args:
        hnm_dir: Path to clean up.
    """
    if os.path.isdir(hnm_dir):
        shutil.rmtree(hnm_dir)


# ---------------------------------------------------------------------------
# Generator and class weight helpers
# ---------------------------------------------------------------------------


def build_generators(
    train_dir: str,
    val_dir: str,
    arch: str,
    seed: int,
) -> Tuple:
    """Create training and validation ImageDataGenerators.

    Training generator applies augmentation matching the baseline protocol.
    Validation generator applies only preprocessing. Neither uses rescale.

    Args:
        train_dir: Path to the training split root.
        val_dir: Path to the validation split root.
        arch: Architecture key for preprocessing function.
        seed: Random seed.

    Returns:
        Tuple (train_gen, val_gen) of DirectoryIterators.
    """
    preprocess_fn = PREPROCESS_FN[arch]

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_fn,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        fill_mode="reflect",
    )

    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_fn,
    )

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=True,
        seed=seed,
    )

    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
        seed=seed,
    )

    return train_gen, val_gen


def compute_class_weights(train_gen) -> Dict[int, float]:
    """Compute balanced class weights from the training generator.

    Args:
        train_gen: DirectoryIterator with a populated classes attribute.

    Returns:
        Dict mapping class index to weight.
    """
    classes_array = train_gen.classes
    unique_classes = np.unique(classes_array)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=classes_array,
    )
    class_weight_dict = dict(zip(unique_classes.tolist(), weights.tolist()))
    print(f"Class weights: {class_weight_dict}")
    return class_weight_dict


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _best_epoch_metrics(history, monitor: str = "val_loss") -> Dict:
    """Extract metrics at the best epoch by monitored value.

    Args:
        history: Keras History object.
        monitor: Metric to minimise.

    Returns:
        Dict of metric name -> value at best epoch.
    """
    hist = history.history
    best_epoch = int(np.argmin(hist[monitor]))
    return {k: hist[k][best_epoch] for k in hist}


def evaluate_val_metrics(
    model: tf.keras.Model,
    val_gen,
) -> Tuple[float, float]:
    """Compute val recall and val PR-AUC for sweep mode.

    Args:
        model: Trained model to evaluate.
        val_gen: Validation generator (shuffle=False).

    Returns:
        Tuple (val_recall, val_pr_auc).
    """
    val_gen.reset()
    preds = model.predict(val_gen, verbose=0).flatten()
    true_labels = val_gen.classes

    # flood=0, non_flood=1 in alphabetical ordering.
    # Model outputs P(non_flood). We need P(flood) for PR-AUC with
    # positive class = flood.
    flood_probs = 1.0 - preds
    flood_true = (true_labels == 0).astype(int)

    # Recall for flood class at threshold 0.5.
    predicted_flood = (flood_probs > 0.5).astype(int)
    true_positives = np.sum((predicted_flood == 1) & (flood_true == 1))
    actual_positives = np.sum(flood_true == 1)
    val_recall = true_positives / actual_positives if actual_positives > 0 else 0.0

    # PR-AUC.
    val_pr_auc = average_precision_score(flood_true, flood_probs)

    return float(val_recall), float(val_pr_auc)


# ---------------------------------------------------------------------------
# Core training routine (shared by percentile, sweep, and no_injection)
# ---------------------------------------------------------------------------


def train_hnm_phases(
    model: tf.keras.Model,
    base_model: tf.keras.Model,
    train_gen,
    val_gen,
    class_weight_dict: Dict[int, float],
    n_trainable: int,
    n_frozen: int,
    output_dir: str,
    log_dir: str,
    arch: str,
    label: str,
    timestamp: str,
) -> tf.keras.Model:
    """Run HNM Phase 1 and Phase 2 training.

    Args:
        model: Full classification model (loaded from checkpoint).
        base_model: Backbone sub-model for freeze control.
        train_gen: Training data generator (HNM-injected or original).
        val_gen: Validation generator (always from ORIGINAL val_dir).
        class_weight_dict: Class weights for the training set.
        n_trainable: Number of layers to unfreeze in Phase 1.
        n_frozen: Number of layers to freeze in Phase 2.
        output_dir: Directory for checkpoints.
        log_dir: Directory for CSV logs.
        arch: Architecture name.
        label: Descriptive label for file naming (e.g. 'hnm_percentile').
        timestamp: Timestamp string for filenames.

    Returns:
        The trained model (with best weights restored by EarlyStopping).
    """
    # -- HNM Phase 1 -------------------------------------------------------
    freeze_for_phase1(base_model, n_trainable=n_trainable)

    phase1_ckpt = os.path.join(output_dir, f"{arch}_{label}_phase1_{timestamp}.keras")
    phase1_log = os.path.join(log_dir, f"{arch}_{label}_phase1_{timestamp}.csv")

    print(f"\n{'='*60}")
    print(f"HNM Phase 1: last {n_trainable} backbone layers trainable, LR={HNM_PHASE1_LR}")
    print(f"  Checkpoint : {phase1_ckpt}")
    print(f"  Log        : {phase1_log}")
    print(f"{'='*60}\n")

    model.compile(
        optimizer=Adam(learning_rate=HNM_PHASE1_LR),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            Precision(name="precision"),
            Recall(name="recall"),
            AUC(name="auc"),
        ],
    )

    phase1_callbacks = build_callbacks(
        checkpoint_path=phase1_ckpt,
        patience=7,
        lr_patience=3,
        log_path=phase1_log,
    )

    history_p1 = model.fit(
        train_gen,
        epochs=HNM_PHASE1_EPOCHS,
        validation_data=val_gen,
        class_weight=class_weight_dict,
        callbacks=phase1_callbacks,
        verbose=1,
    )

    actual_phase1_epochs = len(history_p1.history["loss"])

    # -- HNM Phase 2 -------------------------------------------------------
    freeze_for_phase2(base_model, n_frozen=n_frozen)

    phase2_ckpt = os.path.join(output_dir, f"{arch}_{label}_phase2_{timestamp}.keras")
    phase2_log = os.path.join(log_dir, f"{arch}_{label}_phase2_{timestamp}.csv")

    print(f"\n{'='*60}")
    print(f"HNM Phase 2: freeze first {n_frozen} backbone layers, LR={HNM_PHASE2_LR}")
    print(f"  Checkpoint : {phase2_ckpt}")
    print(f"  Log        : {phase2_log}")
    print(f"{'='*60}\n")

    model.compile(
        optimizer=Adam(learning_rate=HNM_PHASE2_LR),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            Precision(name="precision"),
            Recall(name="recall"),
            AUC(name="auc"),
        ],
    )

    phase2_callbacks = build_callbacks(
        checkpoint_path=phase2_ckpt,
        patience=7,
        lr_patience=4,
        log_path=phase2_log,
    )

    model.fit(
        train_gen,
        epochs=actual_phase1_epochs + HNM_PHASE2_EPOCHS,
        initial_epoch=actual_phase1_epochs,
        validation_data=val_gen,
        class_weight=class_weight_dict,
        callbacks=phase2_callbacks,
        verbose=1,
    )

    # Print summary.
    p1_best = _best_epoch_metrics(history_p1, monitor="val_loss")
    print(f"\n[INFO] Phase 1 best epoch metrics:")
    for k, v in p1_best.items():
        print(f"  {k:<25s} {v:.6f}")

    print(f"\n[INFO] Checkpoints:")
    print(f"  Phase 1 best : {phase1_ckpt}")
    print(f"  Phase 2 best : {phase2_ckpt}")

    return model


# ---------------------------------------------------------------------------
# Read mining candidates
# ---------------------------------------------------------------------------


def read_mining_candidates(results_dir: str, arch: str) -> List[str]:
    """Read mining candidate image paths from the candidates file.

    Args:
        results_dir: Directory containing mining_candidates_{arch}.txt.
        arch: Architecture name.

    Returns:
        List of absolute image paths.

    Raises:
        SystemExit: If file doesn't exist or is empty.
    """
    candidates_path = os.path.join(results_dir, f"mining_candidates_{arch}.txt")

    if not os.path.exists(candidates_path):
        print(
            f"[ERROR] Mining candidates file not found: {candidates_path}\n"
            "Run analyze_confounders.py first."
        )
        sys.exit(1)

    with open(candidates_path, "r", encoding="utf-8") as fh:
        paths = [line.strip() for line in fh if line.strip()]

    if not paths:
        print(
            f"[WARN] Mining candidates file is empty: {candidates_path}\n"
            "No confounder categories exceeded the FP threshold."
        )
        sys.exit(1)

    print(f"[INFO] Loaded {len(paths)} mining candidates from {candidates_path}")
    return paths


# ---------------------------------------------------------------------------
# Log mining scores
# ---------------------------------------------------------------------------


def save_mining_scores(
    scores: List[Tuple[str, float]],
    hard_negative_paths: set,
    log_dir: str,
    arch: str,
) -> None:
    """Save all mining candidate scores to CSV.

    Args:
        scores: List of (path, flood_prob) tuples.
        hard_negative_paths: Set of paths selected as hard negatives.
        log_dir: Directory for log files.
        arch: Architecture name.
    """
    os.makedirs(log_dir, exist_ok=True)
    records = [
        {
            "path": path,
            "flood_prob": flood_prob,
            "is_hard_negative": path in hard_negative_paths,
        }
        for path, flood_prob in scores
    ]
    df = pd.DataFrame(records)
    csv_path = os.path.join(log_dir, f"hnm_candidates_{arch}.csv")
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Mining scores saved to: {csv_path}")


# ---------------------------------------------------------------------------
# Mode: percentile
# ---------------------------------------------------------------------------


def run_percentile_mode(
    args: argparse.Namespace,
    n_trainable: int,
    n_frozen: int,
    timestamp: str,
) -> None:
    """Execute percentile-based hard negative mining and retraining.

    Args:
        args: Parsed CLI arguments.
        n_trainable: Phase 1 trainable layer count.
        n_frozen: Phase 2 frozen layer count.
        timestamp: Timestamp for file naming.
    """
    data_dir = os.path.abspath(args.data_dir)
    output_dir = os.path.abspath(args.output_dir)
    results_dir = os.path.abspath("results")
    log_dir = os.path.join(results_dir, "logs")

    binary_dir = os.path.join(data_dir, "processed_data", "binary")
    train_dir = os.path.join(binary_dir, "train")
    val_dir = os.path.join(binary_dir, "val")
    test_dir = os.path.join(binary_dir, "test")
    hnm_dir = os.path.join(data_dir, "processed_data", "binary_hnm")
    hnm_train_dir = os.path.join(hnm_dir, "train")
    hnm_augmented_dir = os.path.join(hnm_dir, "augmented_hard_negatives")

    # 1. Read candidates.
    candidate_paths = read_mining_candidates(results_dir, args.arch)

    # Partition safety assertion.
    mining_basenames = {os.path.basename(p) for p in candidate_paths}
    verify_partition_safety(mining_basenames, val_dir, test_dir, hnm_augmented_dir)

    # 2. Run mining inference.
    print(f"\n[INFO] Loading baseline model from: {args.model_path}")
    model = tf.keras.models.load_model(args.model_path)

    scores = mine_candidates(model, candidate_paths, args.arch)

    # 3. Sort by flood_prob descending, take top percentage.
    scores.sort(key=lambda x: x[1], reverse=True)
    n_hard = max(1, int(args.top_pct * len(scores)))
    hard_negatives = scores[:n_hard]
    hard_negative_paths_list = [path for path, _ in hard_negatives]

    print(f"\n[INFO] Selected {len(hard_negatives)} hard negatives "
          f"(top {args.top_pct*100:.0f}% of {len(scores)} candidates):")
    for path, prob in hard_negatives:
        print(f"  {os.path.basename(path)}: flood_prob={prob:.4f}")

    # 4. Log all scores.
    hn_set = set(hard_negative_paths_list)
    save_mining_scores(scores, hn_set, log_dir, args.arch)

    # 5. Augment hard negatives.
    augmented_paths = augment_hard_negatives(
        hard_negative_paths_list, hnm_augmented_dir, args.arch
    )

    # 6. Build HNM training directory.
    build_hnm_training_dir(train_dir, hnm_train_dir, augmented_paths)

    # 7. Build generators.
    hnm_train_gen, val_gen = build_generators(
        hnm_train_dir, val_dir, args.arch, args.seed
    )
    print(f"HNM train samples: {hnm_train_gen.samples} | Val samples: {val_gen.samples}")
    print(f"Class indices: {hnm_train_gen.class_indices}")

    # Verify preprocessing.
    verify_preprocessing(hnm_train_gen, args.arch)

    # 8. Compute class weights from HNM training set.
    class_weight_dict = compute_class_weights(hnm_train_gen)

    # 9. Reload model from checkpoint (fresh start for retraining).
    model = tf.keras.models.load_model(args.model_path)
    # Extract base_model (the backbone sub-model).
    base_model = model.layers[1]  # backbone is the second layer after Input

    # 10. Train.
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    model = train_hnm_phases(
        model=model,
        base_model=base_model,
        train_gen=hnm_train_gen,
        val_gen=val_gen,
        class_weight_dict=class_weight_dict,
        n_trainable=n_trainable,
        n_frozen=n_frozen,
        output_dir=output_dir,
        log_dir=log_dir,
        arch=args.arch,
        label="hnm_percentile",
        timestamp=timestamp,
    )

    # 11. Save final model.
    final_path = os.path.join(output_dir, f"{args.arch}_hnm_percentile_{timestamp}.keras")
    model.save(final_path)
    print(f"\n[DONE] HNM percentile model saved: {final_path}")


# ---------------------------------------------------------------------------
# Mode: sweep
# ---------------------------------------------------------------------------


def run_sweep_mode(
    args: argparse.Namespace,
    n_trainable: int,
    n_frozen: int,
    timestamp: str,
) -> None:
    """Execute tau sweep ablation, select best tau, then full retrain.

    Args:
        args: Parsed CLI arguments.
        n_trainable: Phase 1 trainable layer count.
        n_frozen: Phase 2 frozen layer count.
        timestamp: Timestamp for file naming.
    """
    data_dir = os.path.abspath(args.data_dir)
    output_dir = os.path.abspath(args.output_dir)
    results_dir = os.path.abspath("results")
    log_dir = os.path.join(results_dir, "logs")
    tables_dir = os.path.join(results_dir, "tables")

    binary_dir = os.path.join(data_dir, "processed_data", "binary")
    train_dir = os.path.join(binary_dir, "train")
    val_dir = os.path.join(binary_dir, "val")
    test_dir = os.path.join(binary_dir, "test")
    hnm_dir = os.path.join(data_dir, "processed_data", "binary_hnm")
    hnm_train_dir = os.path.join(hnm_dir, "train")
    hnm_augmented_dir = os.path.join(hnm_dir, "augmented_hard_negatives")

    # 1. Read candidates.
    candidate_paths = read_mining_candidates(results_dir, args.arch)

    # Partition safety.
    mining_basenames = {os.path.basename(p) for p in candidate_paths}
    verify_partition_safety(mining_basenames, val_dir, test_dir, hnm_augmented_dir)

    # 2. Score all candidates.
    print(f"\n[INFO] Loading baseline model from: {args.model_path}")
    mining_model = tf.keras.models.load_model(args.model_path)
    scores = mine_candidates(mining_model, candidate_paths, args.arch)
    del mining_model

    # Log all scores.
    save_mining_scores(scores, set(), log_dir, args.arch)

    # 3. Sweep over tau values.
    sweep_results: List[Dict] = []

    print(f"\n{'='*60}")
    print(f"Tau sweep: {SWEEP_TAUS}")
    print(f"{'='*60}\n")

    for tau in SWEEP_TAUS:
        print(f"\n--- Tau = {tau} ---")

        # Select hard negatives by threshold.
        hard_negatives = [(p, fp) for p, fp in scores if fp > tau]
        n_hn = len(hard_negatives)
        print(f"  Hard negatives found: {n_hn}")

        if n_hn == 0:
            print(f"  No hard negatives at tau={tau}. Skipping.")
            sweep_results.append({
                "tau": tau,
                "n_hard_negatives": 0,
                "val_recall": np.nan,
                "val_pr_auc": np.nan,
            })
            continue

        # Augment.
        hard_negative_paths_list = [p for p, _ in hard_negatives]
        temp_aug_dir = os.path.join(hnm_dir, f"temp_aug_tau_{tau}")
        augmented_paths = augment_hard_negatives(
            hard_negative_paths_list, temp_aug_dir, args.arch
        )

        # Build temporary HNM training set.
        temp_train_dir = os.path.join(hnm_dir, f"temp_train_tau_{tau}")
        build_hnm_training_dir(train_dir, temp_train_dir, augmented_paths)

        # Build generators.
        hnm_train_gen, val_gen = build_generators(
            temp_train_dir, val_dir, args.arch, args.seed
        )

        # Reload model from checkpoint.
        model = tf.keras.models.load_model(args.model_path)
        base_model = model.layers[1]

        # Freeze backbone entirely for head-only training.
        base_model.trainable = False

        model.compile(
            optimizer=Adam(learning_rate=HNM_PHASE1_LR),
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                Precision(name="precision"),
                Recall(name="recall"),
                AUC(name="auc"),
            ],
        )

        # Compute class weights.
        cw = compute_class_weights(hnm_train_gen)

        # Train head only for SWEEP_HEAD_EPOCHS (no early stopping).
        model.fit(
            hnm_train_gen,
            epochs=SWEEP_HEAD_EPOCHS,
            validation_data=val_gen,
            class_weight=cw,
            verbose=1,
        )

        # Evaluate val recall and val PR-AUC.
        val_recall, val_pr_auc = evaluate_val_metrics(model, val_gen)
        print(f"  Val recall: {val_recall:.4f}, Val PR-AUC: {val_pr_auc:.4f}")

        sweep_results.append({
            "tau": tau,
            "n_hard_negatives": n_hn,
            "val_recall": val_recall,
            "val_pr_auc": val_pr_auc,
        })

        # Clean up.
        del model
        cleanup_hnm_dir(temp_train_dir)
        cleanup_hnm_dir(temp_aug_dir)

    # 4. Save sweep results.
    os.makedirs(tables_dir, exist_ok=True)
    sweep_df = pd.DataFrame(sweep_results)
    sweep_csv = os.path.join(tables_dir, f"tau_sweep_{args.arch}.csv")
    sweep_df.to_csv(sweep_csv, index=False)

    print(f"\n{'='*60}")
    print("Tau sweep results:")
    print(sweep_df.to_string(index=False))
    print(f"\nSaved to: {sweep_csv}")
    print(f"{'='*60}")

    # 5. Pick best tau: highest val_recall, PR-AUC as tiebreaker.
    valid_results = sweep_df.dropna(subset=["val_recall"])
    if valid_results.empty:
        print("[ERROR] No valid sweep results. No hard negatives found at any tau.")
        sys.exit(1)

    valid_results = valid_results.sort_values(
        ["val_recall", "val_pr_auc"], ascending=[False, False]
    )
    best_row = valid_results.iloc[0]
    best_tau = best_row["tau"]
    print(
        f"\nBest tau: {best_tau} "
        f"(val_recall={best_row['val_recall']:.4f}, "
        f"val_pr_auc={best_row['val_pr_auc']:.4f})"
    )

    # 6. Full retrain with best_tau.
    print(f"\n{'='*60}")
    print(f"Full retrain with best tau = {best_tau}")
    print(f"{'='*60}")

    hard_negatives = [(p, fp) for p, fp in scores if fp > best_tau]
    hard_negative_paths_list = [p for p, _ in hard_negatives]

    print(f"  Hard negatives: {len(hard_negatives)}")
    for path, prob in hard_negatives:
        print(f"    {os.path.basename(path)}: flood_prob={prob:.4f}")

    # Update mining scores with final selection.
    hn_set = set(hard_negative_paths_list)
    save_mining_scores(scores, hn_set, log_dir, args.arch)

    # Augment and build HNM training set.
    cleanup_hnm_dir(hnm_augmented_dir)
    augmented_paths = augment_hard_negatives(
        hard_negative_paths_list, hnm_augmented_dir, args.arch
    )
    build_hnm_training_dir(train_dir, hnm_train_dir, augmented_paths)

    # Build generators.
    hnm_train_gen, val_gen = build_generators(
        hnm_train_dir, val_dir, args.arch, args.seed
    )
    print(f"HNM train samples: {hnm_train_gen.samples} | Val samples: {val_gen.samples}")
    verify_preprocessing(hnm_train_gen, args.arch)

    class_weight_dict = compute_class_weights(hnm_train_gen)

    # Reload model and run full training.
    model = tf.keras.models.load_model(args.model_path)
    base_model = model.layers[1]

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    model = train_hnm_phases(
        model=model,
        base_model=base_model,
        train_gen=hnm_train_gen,
        val_gen=val_gen,
        class_weight_dict=class_weight_dict,
        n_trainable=n_trainable,
        n_frozen=n_frozen,
        output_dir=output_dir,
        log_dir=log_dir,
        arch=args.arch,
        label="hnm_sweep",
        timestamp=timestamp,
    )

    final_path = os.path.join(output_dir, f"{args.arch}_hnm_sweep_{timestamp}.keras")
    model.save(final_path)
    print(f"\n[DONE] HNM sweep model saved: {final_path}")


# ---------------------------------------------------------------------------
# Mode: --no_injection (extended-training control)
# ---------------------------------------------------------------------------


def run_no_injection_mode(
    args: argparse.Namespace,
    n_trainable: int,
    n_frozen: int,
    timestamp: str,
) -> None:
    """Extended-training control: same epoch budget, no HNM injection.

    Args:
        args: Parsed CLI arguments.
        n_trainable: Phase 1 trainable layer count.
        n_frozen: Phase 2 frozen layer count.
        timestamp: Timestamp for file naming.
    """
    data_dir = os.path.abspath(args.data_dir)
    output_dir = os.path.abspath(args.output_dir)
    log_dir = os.path.join("results", "logs")

    binary_dir = os.path.join(data_dir, "processed_data", "binary")
    train_dir = os.path.join(binary_dir, "train")
    val_dir = os.path.join(binary_dir, "val")

    # Build original generators (no HNM injection).
    train_gen, val_gen = build_generators(train_dir, val_dir, args.arch, args.seed)
    print(f"Train samples: {train_gen.samples} | Val samples: {val_gen.samples}")
    print(f"Class indices: {train_gen.class_indices}")

    verify_preprocessing(train_gen, args.arch)

    class_weight_dict = compute_class_weights(train_gen)

    # Load model from same starting weights.
    print(f"\n[INFO] Loading baseline model from: {args.model_path}")
    model = tf.keras.models.load_model(args.model_path)
    base_model = model.layers[1]

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Train for TOTAL_HNM_BUDGET epochs, split into two phases.
    model = train_hnm_phases(
        model=model,
        base_model=base_model,
        train_gen=train_gen,
        val_gen=val_gen,
        class_weight_dict=class_weight_dict,
        n_trainable=n_trainable,
        n_frozen=n_frozen,
        output_dir=output_dir,
        log_dir=log_dir,
        arch=args.arch,
        label="extended_baseline",
        timestamp=timestamp,
    )

    final_path = os.path.join(output_dir, f"{args.arch}_extended_baseline_{timestamp}.keras")
    model.save(final_path)
    print(f"\n[DONE] Extended baseline (no HNM injection) complete.")
    print(f"Model saved: {final_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point: parse args, dispatch to appropriate mode."""
    args = parse_args()
    n_trainable, n_frozen = parse_phase_boundary(args.phase_boundary)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Environment and seed.
    print_runtime_env()
    set_all_seeds(args.seed)
    print(f"Random seed: {args.seed}")

    # Validate model path.
    if not os.path.exists(args.model_path):
        print(f"[ERROR] Model file not found: {args.model_path}")
        sys.exit(1)

    # Validate data directory.
    binary_dir = os.path.join(os.path.abspath(args.data_dir), "processed_data", "binary")
    for subdir in ("train", "val", "test"):
        split_path = os.path.join(binary_dir, subdir)
        if not os.path.isdir(split_path):
            raise FileNotFoundError(
                f"Expected dataset split directory not found: {split_path}\n"
                "Ensure --data_dir points to the dataset root and that "
                "Step 2 (stratified splitting) has been run."
            )

    # Dispatch.
    if args.no_injection:
        print(f"\n{'='*60}")
        print("Mode: Extended-training control (no HNM injection)")
        print(f"Total epoch budget: {TOTAL_HNM_BUDGET}")
        print(f"{'='*60}")
        run_no_injection_mode(args, n_trainable, n_frozen, timestamp)

    elif args.tau_mode == "sweep":
        print(f"\n{'='*60}")
        print("Mode: Tau sweep ablation")
        print(f"Tau values: {SWEEP_TAUS}")
        print(f"{'='*60}")
        run_sweep_mode(args, n_trainable, n_frozen, timestamp)

    else:
        print(f"\n{'='*60}")
        print(f"Mode: Percentile mining (top {args.top_pct*100:.0f}%)")
        print(f"{'='*60}")
        run_percentile_mode(args, n_trainable, n_frozen, timestamp)


if __name__ == "__main__":
    main()
