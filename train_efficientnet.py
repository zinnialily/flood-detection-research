import os
import argparse
import shutil
import numpy as np
from datetime import datetime
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import Precision, Recall, AUC

from sklearn.utils.class_weight import compute_class_weight


# =============================================================================
# ARGUMENTS
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="./data/FloodingDataset2")
parser.add_argument("--output_dir", type=str, default="./models")
args = parser.parse_args()


# =============================================================================
# PATHS & CONSTANTS
# =============================================================================

BASE_DATA = os.path.join(args.data_dir, "processed_data", "binary")
HNM_DATA = os.path.join(args.data_dir, "processed_data", "binary_hnm")

TRAIN_DIR = os.path.join(BASE_DATA, "train")
VAL_DIR = os.path.join(BASE_DATA, "val")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

INITIAL_LR = 1e-4
FINE_TUNE_LR = 1e-5

INITIAL_EPOCHS = 15
FINE_TUNE_EPOCHS = 20
TOTAL_EPOCHS = INITIAL_EPOCHS + FINE_TUNE_EPOCHS

HNM_THRESHOLD = 0.3
HNM_AUG_FACTOR = 5
HNM_LR = 5e-5
HNM_EPOCHS = 15
HNM_FINAL_LR = 1e-5
HNM_FINAL_EPOCHS = 10

tf.random.set_seed(SEED)
np.random.seed(SEED)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(HNM_DATA, exist_ok=True)


# =============================================================================
# DATA GENERATORS
# =============================================================================

train_aug = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    fill_mode="reflect",
)

val_aug = ImageDataGenerator(rescale=1.0 / 255)

train_gen = train_aug.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True,
    seed=SEED,
)

val_gen = val_aug.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False,
)


# =============================================================================
# UTILITIES
# =============================================================================

def compute_weights(generator):
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(generator.classes),
        y=generator.classes,
    )
    return dict(enumerate(weights))


def compile_model(model, lr):
    model.compile(
        optimizer=Adam(lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            Precision(name="precision"),
            Recall(name="recall"),
            AUC(name="auc"),
        ],
    )


def build_callbacks(path, patience, lr_patience):
    return [
        ModelCheckpoint(path, monitor="val_loss", save_best_only=True, verbose=1),
        EarlyStopping(patience=patience, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=lr_patience, min_lr=1e-7, verbose=1),
    ]


# =============================================================================
# MODEL
# =============================================================================

def build_efficientnet():
    base = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
    )

    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    out = Dense(1, activation="sigmoid")(x)

    return Model(base.input, out), base


# =============================================================================
# PHASE 1
# =============================================================================

print("\nPHASE 1 — Partial Unfreezing (last 30 layers trainable)")
model, base_model = build_efficientnet()
class_weights = compute_weights(train_gen)

compile_model(model, INITIAL_LR)

phase1_path = os.path.join(args.output_dir, f"efficientnet_phase1_{timestamp}.keras")

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=INITIAL_EPOCHS,
    class_weight=class_weights,
    callbacks=build_callbacks(phase1_path, patience=7, lr_patience=3),
)


# =============================================================================
# PHASE 2
# =============================================================================

print("\nPHASE 2 — Fine-Tuning (first 50 layers frozen)")

for layer in base_model.layers:
    layer.trainable = True
for layer in base_model.layers[:50]:
    layer.trainable = False

compile_model(model, FINE_TUNE_LR)

phase2_path = os.path.join(args.output_dir, f"efficientnet_phase2_{timestamp}.keras")

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=TOTAL_EPOCHS,
    initial_epoch=INITIAL_EPOCHS,
    class_weight=class_weights,
    callbacks=build_callbacks(phase2_path, patience=10, lr_patience=4),
)

baseline_path = os.path.join(args.output_dir, f"efficientnet_baseline_{timestamp}.keras")
model.save(baseline_path)


# =============================================================================
# HARD NEGATIVE MINING
# =============================================================================

print("\nHARD NEGATIVE MINING")

filepaths = train_gen.filepaths
labels = train_gen.classes

hard_negatives = []

for path, label in tqdm(zip(filepaths, labels), total=len(filepaths)):
    if label != 1:
        continue

    img = load_img(path, target_size=IMG_SIZE)
    x = img_to_array(img) / 255.0
    pred = model.predict(x[None, ...], verbose=0)[0][0]

    flood_prob = 1.0 - pred
    if flood_prob > 0.7:  # identical behavior to original logic
        hard_negatives.append(path)

print(f"Hard negatives found: {len(hard_negatives)}")


# =============================================================================
# CREATE ENHANCED DATASET
# =============================================================================

hnm_train_dir = os.path.join(HNM_DATA, "train")

for cls in ["flood", "non_flood"]:
    os.makedirs(os.path.join(hnm_train_dir, cls), exist_ok=True)

for path, label in zip(filepaths, labels):
    cls = "flood" if label == 0 else "non_flood"
    shutil.copy2(path, os.path.join(hnm_train_dir, cls, os.path.basename(path)))

strong_aug = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.3,
    brightness_range=[0.6, 1.4],
    horizontal_flip=True,
)

for path in tqdm(hard_negatives):
    img = load_img(path, target_size=IMG_SIZE)
    x = img_to_array(img)
    base_name, ext = os.path.splitext(os.path.basename(path))

    for i in range(HNM_AUG_FACTOR):
        aug = strong_aug.random_transform(x)
        fname = f"{base_name}_hnm_aug_{i}{ext}"
        tf.keras.preprocessing.image.save_img(
            os.path.join(hnm_train_dir, "non_flood", fname),
            aug,
        )


# =============================================================================
# HNM RETRAINING
# =============================================================================

hnm_train_gen = train_aug.flow_from_directory(
    hnm_train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True,
    seed=SEED,
)

hnm_weights = compute_weights(hnm_train_gen)

compile_model(model, HNM_LR)

hnm_path = os.path.join(args.output_dir, f"efficientnet_hnm_{timestamp}.keras")

model.fit(
    hnm_train_gen,
    validation_data=val_gen,
    epochs=HNM_EPOCHS,
    class_weight=hnm_weights,
    callbacks=build_callbacks(hnm_path, patience=7, lr_patience=3),
)


# =============================================================================
# FINAL FINE-TUNING
# =============================================================================

for layer in base_model.layers:
    layer.trainable = True

compile_model(model, HNM_FINAL_LR)

hnm_final_path = os.path.join(args.output_dir, f"efficientnet_hnm_final_{timestamp}.keras")

model.fit(
    hnm_train_gen,
    validation_data=val_gen,
    epochs=HNM_EPOCHS + HNM_FINAL_EPOCHS,
    initial_epoch=HNM_EPOCHS,
    class_weight=hnm_weights,
    callbacks=build_callbacks(hnm_final_path, patience=10, lr_patience=4),
)

final_model_path = os.path.join(args.output_dir, f"efficientnet_hnm_complete_{timestamp}.keras")
model.save(final_model_path)

print("\nTRAINING COMPLETE")
