import os
import argparse
import numpy as np
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    CSVLogger,
    TensorBoard,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import Precision, Recall, AUC

from sklearn.utils.class_weight import compute_class_weight


# =============================================================================
# ARGUMENTS
# =============================================================================

parser = argparse.ArgumentParser(description="Train ResNet50 for Flood Detection")
parser.add_argument("--data_dir", type=str, default="./data/FloodingDataset2")
parser.add_argument("--output_dir", type=str, default="./models/resnet_50")
args = parser.parse_args()


# =============================================================================
# CONFIGURATION
# =============================================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
INITIAL_EPOCHS = 15
FINE_TUNE_EPOCHS = 20
INITIAL_LR = 1e-4
FINE_TUNE_LR = 1e-5
RANDOM_SEED = 42

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DATA_PATH = os.path.join(args.data_dir, "processed_data", "binary")
TRAIN_DIR = os.path.join(DATA_PATH, "train")
VAL_DIR = os.path.join(DATA_PATH, "val")

MODEL_DIR = args.output_dir
LOG_DIR = os.path.join(MODEL_DIR, "logs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

print("Training ResNet50 Binary Flood Detection Model")
print(f"Dataset: {args.data_dir}")
print(f"Output: {MODEL_DIR}")


# =============================================================================
# DATA GENERATORS
# =============================================================================

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    fill_mode="reflect",
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True,
    seed=RANDOM_SEED,
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False,
)


# =============================================================================
# CLASS WEIGHTS
# =============================================================================

def compute_class_weights(generator):
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(generator.classes),
        y=generator.classes,
    )
    return dict(enumerate(weights))


class_weight_dict = compute_class_weights(train_generator)


# =============================================================================
# MODEL BUILDING
# =============================================================================

def build_model():
    base = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    )

    # Phase 1 setup: last 30 layers trainable
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    x = base.output
    x = GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    return Model(inputs=base.input, outputs=outputs), base


def compile_model(model, lr):
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            Precision(name="precision"),
            Recall(name="recall"),
            AUC(name="auc"),
        ],
    )


def build_callbacks(phase_name, patience, lr_patience):
    return [
        ModelCheckpoint(
            os.path.join(MODEL_DIR, f"resnet50_{phase_name}_best_{timestamp}.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=lr_patience,
            min_lr=1e-7,
            verbose=1,
        ),
        CSVLogger(os.path.join(LOG_DIR, f"train_{phase_name}_{timestamp}.csv")),
        TensorBoard(log_dir=os.path.join(LOG_DIR, f"tb_{phase_name}_{timestamp}")),
    ]


# =============================================================================
# PHASE 1 TRAINING (15 EPOCHS)
# =============================================================================

model, base_model = build_model()
compile_model(model, INITIAL_LR)

model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=INITIAL_EPOCHS,
    class_weight=class_weight_dict,
    callbacks=build_callbacks("phase1", patience=7, lr_patience=3),
    verbose=1,
)


# =============================================================================
# PHASE 2 FINE-TUNING (20 EPOCHS)
# =============================================================================

for layer in base_model.layers:
    layer.trainable = True

for layer in base_model.layers[:50]:
    layer.trainable = False

compile_model(model, FINE_TUNE_LR)

model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
    initial_epoch=INITIAL_EPOCHS,
    class_weight=class_weight_dict,
    callbacks=build_callbacks("phase2", patience=10, lr_patience=4),
    verbose=1,
)


# =============================================================================
# SAVE FINAL MODEL
# =============================================================================

final_model_path = os.path.join(MODEL_DIR, "resnet50.keras")
model.save(final_model_path)

print("Training complete.")
print(f"Final model saved to: {final_model_path}")
