"""
utils.py -- Shared utilities for the flood detection study.

Single source of truth for:
  - Random seed initialisation
  - Preprocessing functions (Option B: explicit preprocessing_function in generator,
    include_preprocessing=False on backbone -- fixes the double-rescaling bug)
  - Model building and layer-freeze helpers
  - Training callbacks factory
"""

import os
import random
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, ResNet50
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
)
from tensorflow.keras.models import Model


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

PREPROCESS_FN = {
    "efficientnet": tf.keras.applications.efficientnet.preprocess_input,
    "resnet50": tf.keras.applications.resnet50.preprocess_input,
}
"""
Preprocessing functions keyed by architecture name.

Both functions expect raw [0, 255] uint8 or float32 pixel input.
Use as ``preprocessing_function=PREPROCESS_FN[arch]`` in ImageDataGenerator.
Do NOT combine with ``rescale=1./255`` -- that would re-introduce the
double-rescaling bug this project is correcting.

Expected output ranges after preprocessing:
  efficientnet: maps [0, 255] -> [-1, 1]  (tf.keras.applications.efficientnet)
  resnet50:     subtracts ImageNet channel means, output roughly [-123, 151]
"""


# ---------------------------------------------------------------------------
# Seed initialisation
# ---------------------------------------------------------------------------


def set_all_seeds(seed: int) -> None:
    """Seed all randomness sources for reproducibility.

    Must be called at the top of every training script/notebook before any
    library that uses random state (NumPy, TensorFlow, Python random, hash
    randomisation).

    Args:
        seed: Integer seed value (e.g. 42, 123, 256, 512, 1024).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ---------------------------------------------------------------------------
# Preprocessing verification
# ---------------------------------------------------------------------------


def verify_preprocessing(generator, arch: str) -> None:
    """Assert that a data generator is producing correctly preprocessed batches.

    Grabs one batch from the generator and checks that pixel values fall in the
    range expected by the given backbone architecture.  Raises ``AssertionError``
    immediately if the range is wrong so training is halted before a silent
    double-rescaling bug contaminates results.

    Args:
        generator: A Keras ``ImageDataGenerator``-backed iterator or any
            generator that yields ``(batch_images, batch_labels)`` tuples.
        arch: One of ``"efficientnet"`` or ``"resnet50"``.

    Raises:
        AssertionError: If the batch statistics do not match the expected range
            for the chosen architecture.
        ValueError: If ``arch`` is not a recognised key.
    """
    if arch not in PREPROCESS_FN:
        raise ValueError(
            f"Unknown arch '{arch}'. Valid choices: {list(PREPROCESS_FN.keys())}"
        )

    batch, _ = next(iter(generator))
    batch_min = float(batch.min())
    batch_max = float(batch.max())

    if arch == "resnet50":
        # resnet50.preprocess_input subtracts ImageNet channel means
        # (103.939, 116.779, 123.68) from [0,255] input, so the minimum
        # value can reach roughly -123.  Values > 0 are expected for bright
        # pixels.  A min > -50 strongly indicates inputs were pre-scaled to
        # [0,1] before preprocessing, which is the double-rescaling bug.
        assert batch_min < -50, (
            f"[FAIL] ResNet50 preprocessing check: batch.min()={batch_min:.3f}, "
            "expected < -50.  "
            "Fix: remove 'rescale=1./255' from your ImageDataGenerator and pass "
            "raw [0, 255] pixels with "
            "preprocessing_function=PREPROCESS_FN['resnet50']."
        )

    elif arch == "efficientnet":
        # efficientnet.preprocess_input maps [0, 255] -> [-1, 1].
        # If the generator is already rescaling to [0, 1], the preprocessing
        # function maps [0, 1] -> roughly [-1, -0.992], so batch.max() would
        # be close to -0.992 rather than +1.  Checking both conditions catches
        # both the double-rescale bug and a missing preprocessing_function.
        assert batch_min < 0 and batch_max <= 1.01, (
            f"[FAIL] EfficientNet preprocessing check: "
            f"batch range=[{batch_min:.3f}, {batch_max:.3f}], "
            "expected min < 0 and max <= 1.01 (i.e. roughly [-1, 1]).  "
            "Fix: remove 'rescale=1./255' from your ImageDataGenerator and pass "
            "raw [0, 255] pixels with "
            "preprocessing_function=PREPROCESS_FN['efficientnet']."
        )

    print(
        f"[OK] Preprocessing verified for {arch}: "
        f"range=[{batch_min:.3f}, {batch_max:.3f}]"
    )


# ---------------------------------------------------------------------------
# Layer-freeze helpers
# ---------------------------------------------------------------------------


def freeze_for_phase1(base_model: Model, n_trainable: int = 30) -> None:
    """Configure the backbone for Phase 1 partial fine-tuning.

    Makes the backbone trainable, then freezes all layers except the last
    ``n_trainable`` layers.  The unfrozen layers receive gradient updates
    while deeper pretrained features are preserved.

    Args:
        base_model: The backbone ``Model`` (EfficientNetB0 or ResNet50).
        n_trainable: Number of layers from the end of the backbone to leave
            trainable.  Default 30 (as per project hyperparameter table).
    """
    base_model.trainable = True
    freeze_cutoff = len(base_model.layers) - n_trainable
    for layer in base_model.layers[:freeze_cutoff]:
        layer.trainable = False


def freeze_for_phase2(base_model: Model, n_frozen: int = 50) -> None:
    """Configure the backbone for Phase 2 extended fine-tuning.

    Makes all layers trainable, then re-freezes the first ``n_frozen`` layers
    so that only the earliest (most general) pretrained features are protected.

    Args:
        base_model: The backbone ``Model`` (EfficientNetB0 or ResNet50).
        n_frozen: Number of layers from the start of the backbone to keep
            frozen.  Default 50 (as per project hyperparameter table).
    """
    base_model.trainable = True
    for layer in base_model.layers[:n_frozen]:
        layer.trainable = False


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------


def build_model(
    arch: str,
    phase_boundary: Tuple[int, int] = (30, 50),
) -> Tuple[Model, Model]:
    """Build the classification model for the given backbone architecture.

    Both backbones are instantiated with ``include_preprocessing=False``
    (Option B of the project's double-rescaling fix).  All preprocessing is
    delegated to the data generator via ``PREPROCESS_FN``.

    The classification head is identical for both architectures:
        GlobalAveragePooling2D
        -> Dropout(0.2)
        -> Dense(256, relu)
        -> BatchNormalization
        -> Dropout(0.3)
        -> Dense(1, sigmoid)

    Phase 1 freezing is applied by default via ``freeze_for_phase1``.

    Args:
        arch: One of ``"efficientnet"`` or ``"resnet50"``.
        phase_boundary: Tuple ``(n_trainable_phase1, n_frozen_phase2)``.
            First element is passed to ``freeze_for_phase1``; second is
            available for the caller to pass to ``freeze_for_phase2`` at
            Phase 2 transition.  Default ``(30, 50)``.

    Returns:
        A tuple ``(full_model, base_model)`` where ``full_model`` is the
        end-to-end Keras ``Model`` ready for compilation and ``base_model``
        is the backbone sub-model (useful for later calls to
        ``freeze_for_phase2``).

    Raises:
        ValueError: If ``arch`` is not a recognised key.
    """
    if arch not in PREPROCESS_FN:
        raise ValueError(
            f"Unknown arch '{arch}'. Valid choices: {list(PREPROCESS_FN.keys())}"
        )

    inputs = tf.keras.Input(shape=(224, 224, 3), name="input_image")

    if arch == "efficientnet":
        base_model = EfficientNetB0(
            weights="imagenet",
            include_top=False,
            include_preprocessing=False,
            input_shape=(224, 224, 3),
        )
    else:  # resnet50
        base_model = ResNet50(
            weights="imagenet",
            include_top=False,
            include_preprocessing=False,
            input_shape=(224, 224, 3),
        )

    # Apply Phase 1 freezing immediately after construction.
    freeze_for_phase1(base_model, n_trainable=phase_boundary[0])

    # Build the classification head.
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D(name="gap")(x)
    x = Dropout(0.2, name="dropout_1")(x)
    x = Dense(256, activation="relu", name="dense_256")(x)
    x = BatchNormalization(name="bn")(x)
    x = Dropout(0.3, name="dropout_2")(x)
    outputs = Dense(1, activation="sigmoid", name="output")(x)

    full_model = Model(inputs=inputs, outputs=outputs, name=f"{arch}_flood_detector")

    return full_model, base_model


# ---------------------------------------------------------------------------
# Callbacks factory
# ---------------------------------------------------------------------------


def build_callbacks(
    checkpoint_path: str,
    patience: int,
    lr_patience: int,
    log_path: str,
) -> List:
    """Create the standard set of training callbacks.

    Args:
        checkpoint_path: File path at which the best model weights are saved
            (e.g. ``"/content/drive/MyDrive/.../efficientnet_phase1_best.keras"``).
        patience: Number of epochs without ``val_loss`` improvement before
            ``EarlyStopping`` terminates training.
        lr_patience: Number of epochs without ``val_loss`` improvement before
            ``ReduceLROnPlateau`` halves the learning rate.
        log_path: File path for the ``CSVLogger`` output
            (e.g. ``"results/logs/efficientnet_phase1.csv"``).

    Returns:
        A list containing:
          [ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger]
    """
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=lr_patience,
        min_lr=1e-7,
        verbose=1,
    )

    csv_logger = CSVLogger(filename=log_path)

    return [checkpoint, early_stopping, reduce_lr, csv_logger]
