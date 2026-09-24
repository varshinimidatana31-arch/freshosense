"""
MODULE 1 - Food Identification
==============================

Identifies what food item is in an image captured by the refrigerator's
internal camera or the mobile app.

Approach: transfer learning on MobileNetV2 (ImageNet weights). MobileNetV2 is
chosen because it is small enough (~14 MB) to run on the fridge's embedded
board or on a phone via TensorFlow Lite.

Dataset layout expected for training:

    data/food/
        train/
            apple/    img1.jpg img2.jpg ...
            banana/   ...
            tomato/   ...
        val/
            apple/    ...
            banana/   ...

Usage:
    python food_identification.py --train --data data/food
    python food_identification.py --predict test.jpg
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

import config


# ----------------------------------------------------------------------------
# Result container
# ----------------------------------------------------------------------------
@dataclass
class FoodPrediction:
    label: str
    confidence: float
    top_k: list[tuple[str, float]]

    @property
    def is_confident(self) -> bool:
        return self.confidence >= config.MIN_IDENTIFICATION_CONFIDENCE

    @property
    def is_raw_food(self) -> bool:
        return self.label in config.RAW_FOOD_CLASSES

    def __str__(self) -> str:
        return f"{self.label} ({self.confidence * 100:.1f}%)"


# ----------------------------------------------------------------------------
# Data pipeline
# ----------------------------------------------------------------------------
def _load_datasets(data_dir: Path):
    """Build train/validation tf.data pipelines from the folder structure."""
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir / "train",
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        label_mode="categorical",
        shuffle=True,
        seed=config.SEED,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir / "val",
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        label_mode="categorical",
        shuffle=False,
    )
    class_names = train_ds.class_names

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    return train_ds, val_ds, class_names


def _augmentation_block() -> tf.keras.Sequential:
    """Fridge photos vary in angle and lighting, so augment for that."""
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.15),
            layers.RandomZoom(0.15),
            layers.RandomContrast(0.20),
            layers.RandomBrightness(0.20),
        ],
        name="augmentation",
    )


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def build_model(num_classes: int) -> tf.keras.Model:
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*config.IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False  # frozen for stage 1

    inputs = layers.Input(shape=(*config.IMG_SIZE, 3), name="image")
    x = _augmentation_block()(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="food")(x)

    model = models.Model(inputs, outputs, name="food_identifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(data_dir: str | Path, epochs: int = 15, fine_tune_epochs: int = 8):
    """Two-stage training: frozen backbone, then fine-tune the top layers."""
    data_dir = Path(data_dir)
    train_ds, val_ds, class_names = _load_datasets(data_dir)
    print(f"Classes found ({len(class_names)}): {class_names}")

    model = build_model(len(class_names))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=4, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=2, min_lr=1e-6
        ),
    ]

    print("\n--- Stage 1: training classifier head ---")
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)

    print("\n--- Stage 2: fine-tuning MobileNetV2 top layers ---")
    base = model.get_layer("mobilenetv2_1.00_224")
    base.trainable = True
    for layer in base.layers[:-30]:   # unfreeze only the last 30 layers
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds, validation_data=val_ds, epochs=fine_tune_epochs, callbacks=callbacks
    )

    loss, acc = model.evaluate(val_ds)
    print(f"\nFinal validation accuracy: {acc * 100:.2f}%  (loss {loss:.4f})")

    model.save(config.FOOD_MODEL_PATH)
    config.FOOD_LABELS_PATH.write_text(json.dumps(class_names, indent=2))
    print(f"Saved model  -> {config.FOOD_MODEL_PATH}")
    print(f"Saved labels -> {config.FOOD_LABELS_PATH}")
    return model, class_names


# ----------------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------------
class FoodIdentifier:
    """Loads the trained model once and answers prediction requests."""

    def __init__(self, model_path: Path | None = None, labels_path: Path | None = None):
        model_path = Path(model_path or config.FOOD_MODEL_PATH)
        labels_path = Path(labels_path or config.FOOD_LABELS_PATH)

        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained model at {model_path}. "
                f"Run:  python food_identification.py --train --data data/food"
            )

        self.model = tf.keras.models.load_model(model_path)
        self.labels = (
            json.loads(labels_path.read_text())
            if labels_path.exists()
            else config.FOOD_CLASSES
        )

    @staticmethod
    def _load_image(image_path: str | Path) -> np.ndarray:
        img = tf.keras.utils.load_img(image_path, target_size=config.IMG_SIZE)
        arr = tf.keras.utils.img_to_array(img)
        return np.expand_dims(arr, axis=0)

    def predict(self, image_path: str | Path, top_k: int = 3) -> FoodPrediction:
        batch = self._load_image(image_path)
        probs = self.model.predict(batch, verbose=0)[0]

        order = np.argsort(probs)[::-1][:top_k]
        ranked = [(self.labels[i], float(probs[i])) for i in order]

        return FoodPrediction(
            label=ranked[0][0], confidence=ranked[0][1], top_k=ranked
        )

    def predict_batch(self, image_paths: list[str | Path]) -> list[FoodPrediction]:
        """Used when the internal camera captures a full shelf of items."""
        return [self.predict(p) for p in image_paths]

    def export_tflite(self, out_path: str | Path = "models/food_identifier.tflite"):
        """Quantized export for the fridge board / mobile app."""
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        Path(out_path).write_bytes(converter.convert())
        print(f"TFLite model written to {out_path}")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ZeroWaste AI - food identification")
    parser.add_argument("--train", action="store_true", help="train the model")
    parser.add_argument("--data", default="data/food", help="dataset root")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--predict", metavar="IMAGE", help="classify one image")
    args = parser.parse_args()

    if args.train:
        train(args.data, epochs=args.epochs)
    elif args.predict:
        result = FoodIdentifier().predict(args.predict)
        print(f"\nIdentified: {result}")
        print("Top matches:")
        for name, score in result.top_k:
            print(f"   {name:<15} {score * 100:5.1f}%")
        if not result.is_confident:
            print("\nLow confidence - ask the user to confirm the item name.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
