"""
train_baseline.py -- Two-phase baseline training for EfficientNetB0 or ResNet50.

Replaces the old train_efficientnet.py and train_resnet50.py with a single
unified script that supports both architectures via --arch.

Usage:
    python scripts/train_baseline.py --arch efficientnet
    python scripts/train_baseline.py --arch resnet50 --seed 123

Compute: Colab T4 GPU (or any CUDA-capable GPU).
Estimated runtime: ~1-2 hours per architecture on T4.
Dependencies: tensorflow>=2.16, scikit-learn, numpy.
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------------------------------------------------------------------
# Path setup: allow running as a top-level script (python scripts/train_baseline.py)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    PREPROCESS_FN,
    build_callbacks,
    build_model,
    freeze_for_phase2,
    set_all_seeds,
    verify_preprocessing,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BATCH_SIZE: int = 32
IMG_SIZE: tuple[int, int] = (224, 224)
PHASE1_LR: float = 1e-4
PHASE2_LR: float = 1e-5
PHASE1_MAX_EPOCHS: int = 15
PHASE2_MAX_EPOCHS: int = 20
EARLY_STOPPING_PATIENCE: int = 7
LR_PATIENCE_PHASE1: int = 3
LR_PATIENCE_PHASE2: int = 4


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train a two-phase baseline flood detection model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--arch",
        required=True,
        choices=["efficientnet", "resnet50"],
        help="Backbone architecture to train.",
    )
    parser.add_argument(
        "--data_dir",
        default="./data/FloodingDataset2",
        help="Path to the dataset root directory.",
    )
    parser.add_argument(
        "--output_dir",
        default="./models",
        help="Directory where model checkpoints are saved.",
    )
    parser.add_argument(
        "--phase_boundary",
        default="30,50",
        help=(
            "Comma-separated pair n_trainable,n_frozen defining Phase 1 and "
            "Phase 2 freeze boundaries (e.g. '30,50')."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def parse_phase_boundary(value: str) -> tuple[int, int]:
    """Parse a 'n_trainable,n_frozen' string into a tuple of ints.

    Args:
        value: Comma-separated string with exactly two positive integers.

    Returns:
        Tuple ``(n_trainable, n_frozen)``.

    Raises:
        argparse.ArgumentTypeError: If the string cannot be parsed or values
            are not positive integers.
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
    result = subprocess.run(
        ["nvidia-smi"], capture_output=True, text=True
    )
    if result.returncode == 0:
        print(result.stdout[:600])
    else:
        print("nvidia-smi not available (CPU-only runtime or no driver).")
    print(f"TensorFlow version : {tf.__version__}")
    print(f"Python version     : {sys.version}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Data generator factory
# ---------------------------------------------------------------------------


def build_generators(
    train_dir: str,
    val_dir: str,
    arch: str,
    seed: int,
) -> tuple:
    """Create training and validation ImageDataGenerators.

    Training generator applies augmentation (rotation, shifts, flip, zoom,
    brightness).  Validation generator applies only the backbone preprocessing
    function.  Neither generator uses 'rescale' -- all normalisation is handled
    by ``PREPROCESS_FN[arch]`` to avoid the double-rescaling bug.

    Args:
        train_dir: Path to the train split root (contains flood/ and non_flood/).
        val_dir: Path to the val split root.
        arch: Architecture key used to select the preprocessing function.
        seed: Random seed for shuffling and augmentation.

    Returns:
        Tuple ``(train_gen, val_gen)`` of DirectoryIterators.
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
        # NOTE: no rescale -- preprocessing_function handles normalisation.
    )

    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_fn,
        # No augmentation, no rescale.
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


# ---------------------------------------------------------------------------
# Class weight computation
# ---------------------------------------------------------------------------


def compute_class_weights(train_gen) -> dict[int, float]:
    """Compute balanced class weights from the training generator's label array.

    Args:
        train_gen: A DirectoryIterator with a populated ``classes`` attribute.

    Returns:
        Dictionary mapping class index to weight, e.g. ``{0: 1.2, 1: 0.85}``.
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
# Metric helpers for post-training summary
# ---------------------------------------------------------------------------


def _best_epoch_metrics(history, monitor: str = "val_loss") -> dict:
    """Extract metrics at the epoch with the best monitored value.

    Args:
        history: Keras History object returned by model.fit().
        monitor: Metric name to minimise (``val_loss``) when selecting the
            best epoch.

    Returns:
        Dictionary of metric name -> value at the best epoch.
    """
    hist = history.history
    best_epoch = int(np.argmin(hist[monitor]))
    return {k: hist[k][best_epoch] for k in hist}


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point: parse args, build data, train Phase 1 + Phase 2."""
    args = parse_args()
    n_trainable, n_frozen = parse_phase_boundary(args.phase_boundary)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # -- Environment --------------------------------------------------------
    print_runtime_env()

    # -- Reproducibility ----------------------------------------------------
    set_all_seeds(args.seed)
    print(f"Random seed: {args.seed}")

    # -- Directory setup ----------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.join("results", "logs")
    os.makedirs(log_dir, exist_ok=True)

    # -- Data paths ---------------------------------------------------------
    binary_root = os.path.join(
        args.data_dir, "processed_data", "binary"
    )
    train_dir = os.path.join(binary_root, "train")
    val_dir = os.path.join(binary_root, "val")

    for split_dir in (train_dir, val_dir):
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"Expected dataset split directory not found: {split_dir}\n"
                "Ensure --data_dir points to the dataset root and that "
                "Step 2 (stratified splitting) has been run."
            )

    # -- Generators ---------------------------------------------------------
    print("\nBuilding data generators ...")
    train_gen, val_gen = build_generators(
        train_dir=train_dir,
        val_dir=val_dir,
        arch=args.arch,
        seed=args.seed,
    )
    print(
        f"Train samples: {train_gen.samples} | "
        f"Val samples: {val_gen.samples}"
    )
    print(f"Class indices: {train_gen.class_indices}")

    # -- Preprocessing verification (halts on failure) ----------------------
    print("\nVerifying preprocessing ...")
    verify_preprocessing(train_gen, args.arch)

    # -- Class weights ------------------------------------------------------
    class_weight_dict = compute_class_weights(train_gen)

    # -- Model --------------------------------------------------------------
    print(f"\nBuilding model: arch={args.arch}, phase_boundary=({n_trainable}, {n_frozen})")
    model, base_model = build_model(args.arch, (n_trainable, n_frozen))
    model.summary(print_fn=print)

    # -- Phase 1 ------------------------------------------------------------
    phase1_ckpt = os.path.join(
        args.output_dir, f"{args.arch}_phase1_{timestamp}.keras"
    )
    phase1_log = os.path.join(log_dir, f"{args.arch}_baseline_phase1_{timestamp}.csv")

    print(f"\n{'='*60}")
    print(f"Phase 1: last {n_trainable} backbone layers trainable, LR={PHASE1_LR}")
    print(f"  Checkpoint : {phase1_ckpt}")
    print(f"  Log        : {phase1_log}")
    print(f"{'='*60}\n")

    model.compile(
        optimizer=Adam(learning_rate=PHASE1_LR),
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
        patience=EARLY_STOPPING_PATIENCE,
        lr_patience=LR_PATIENCE_PHASE1,
        log_path=phase1_log,
    )

    history_p1 = model.fit(
        train_gen,
        epochs=PHASE1_MAX_EPOCHS,
        validation_data=val_gen,
        class_weight=class_weight_dict,
        callbacks=phase1_callbacks,
        verbose=1,
    )

    # -- Phase 2 ------------------------------------------------------------
    phase2_ckpt = os.path.join(
        args.output_dir, f"{args.arch}_phase2_{timestamp}.keras"
    )
    phase2_log = os.path.join(log_dir, f"{args.arch}_baseline_phase2_{timestamp}.csv")

    print(f"\n{'='*60}")
    print(f"Phase 2: freeze first {n_frozen} backbone layers, LR={PHASE2_LR}")
    print(f"  Checkpoint : {phase2_ckpt}")
    print(f"  Log        : {phase2_log}")
    print(f"{'='*60}\n")

    freeze_for_phase2(base_model, n_frozen=n_frozen)

    model.compile(
        optimizer=Adam(learning_rate=PHASE2_LR),
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
        patience=EARLY_STOPPING_PATIENCE,
        lr_patience=LR_PATIENCE_PHASE2,
        log_path=phase2_log,
    )

    actual_phase1_epochs = len(history_p1.history["loss"])

    history_p2 = model.fit(
        train_gen,
        epochs=actual_phase1_epochs + PHASE2_MAX_EPOCHS,
        initial_epoch=actual_phase1_epochs,
        validation_data=val_gen,
        class_weight=class_weight_dict,
        callbacks=phase2_callbacks,
        verbose=1,
    )

    # -- Save final model ---------------------------------------------------
    final_path = os.path.join(
        args.output_dir, f"{args.arch}_baseline_{timestamp}.keras"
    )
    model.save(final_path)
    print(f"\nFinal model saved: {final_path}")

    # -- Summary ------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Training summary")
    print(f"{'='*60}")

    p1_best = _best_epoch_metrics(history_p1, monitor="val_loss")
    print("\nPhase 1 best epoch metrics (by val_loss):")
    for k, v in p1_best.items():
        print(f"  {k:<25s} {v:.6f}")

    p2_best = _best_epoch_metrics(history_p2, monitor="val_loss")
    print("\nPhase 2 best epoch metrics (by val_loss):")
    for k, v in p2_best.items():
        print(f"  {k:<25s} {v:.6f}")

    print(f"\nCheckpoints:")
    print(f"  Phase 1 best : {phase1_ckpt}")
    print(f"  Phase 2 best : {phase2_ckpt}")
    print(f"  Final        : {final_path}")
    print(f"Logs:")
    print(f"  Phase 1      : {phase1_log}")
    print(f"  Phase 2      : {phase2_log}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
