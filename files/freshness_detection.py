"""
MODULE 3 - Raw Food Detection / Freshness Prediction
====================================================

Given an image of a raw item (fruit, vegetable, meat, fish), this module
predicts whether it is fresh or spoiling and returns a 0-100 freshness score
plus an estimate of how many days of usable life are left.

It combines two signals:

  1. A CNN (MobileNetV2 transfer learning) trained on fresh/rotten images.
  2. Classical OpenCV features - brown/dark-spot ratio, colour saturation,
     and surface texture roughness - which catch early spoilage the CNN
     sometimes misses and make the output explainable to the user.

The two are fused with a weighted average. The CV features also let the
module degrade gracefully: if no CNN has been trained yet, it still returns
a usable score from image analysis alone.

Dataset layout for training:

    data/freshness/
        train/
            fresh/    ...
            rotten/   ...
        val/
            fresh/    ...
            rotten/   ...

Usage:
    python freshness_detection.py --train --data data/freshness
    python freshness_detection.py --predict tomato.jpg --food tomato
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

import config

# How much each signal contributes to the final score
CNN_WEIGHT = 0.65
CV_WEIGHT = 0.35


# ----------------------------------------------------------------------------
# Result container
# ----------------------------------------------------------------------------
@dataclass
class FreshnessResult:
    food_name: str
    score: float                 # 0-100, higher = fresher
    status: str                  # FRESH / USE_SOON / DEGRADING / SPOILED
    advice: str
    estimated_days_left: int
    evidence: dict = field(default_factory=dict)

    @property
    def is_safe_to_eat(self) -> bool:
        return self.status != "SPOILED"

    def __str__(self) -> str:
        return (
            f"{self.food_name}: {self.status} "
            f"(score {self.score:.0f}/100, ~{self.estimated_days_left} day(s) left)"
        )


def _band_for(score: float) -> tuple[str, str]:
    for threshold, status, advice in config.FRESHNESS_BANDS:
        if score >= threshold:
            return status, advice
    return "SPOILED", "Do not consume. Discard this item."


# ----------------------------------------------------------------------------
# Signal 2: classical computer-vision features
# ----------------------------------------------------------------------------
def _segment_food(bgr: np.ndarray) -> np.ndarray:
    """
    Separate the food object from the fridge background using Otsu
    thresholding on the saturation channel. Returns a binary mask.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    _, mask = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # If segmentation fails badly, fall back to using the whole frame
    if cv2.countNonZero(mask) < 0.05 * mask.size:
        mask = np.full(mask.shape, 255, dtype=np.uint8)
    return mask


def extract_cv_features(image_path: str | Path) -> dict:
    """
    Returns interpretable spoilage indicators:

      dark_spot_ratio   fraction of the food surface that is brown/black
      mean_saturation   colour vividness (fresh produce is saturated)
      texture_energy    surface roughness via Laplacian variance
                        (wrinkling and mould raise it)
      mould_ratio       fraction matching fuzzy white/grey-green mould tones
    """
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    bgr = cv2.resize(bgr, config.IMG_SIZE)
    mask = _segment_food(bgr)
    food_pixels = cv2.countNonZero(mask)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # --- brown / black decay spots -----------------------------------------
    brown = cv2.inRange(hsv, np.array([5, 40, 20]), np.array([25, 255, 120]))
    black = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 55]))
    decay = cv2.bitwise_and(cv2.bitwise_or(brown, black), mask)
    dark_spot_ratio = cv2.countNonZero(decay) / max(food_pixels, 1)

    # --- fuzzy mould (low saturation, high value patches) -------------------
    mould = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 45, 255]))
    mould = cv2.bitwise_and(mould, mask)
    mould_ratio = cv2.countNonZero(mould) / max(food_pixels, 1)

    # --- colour vividness ---------------------------------------------------
    mean_saturation = float(cv2.mean(s, mask=mask)[0])

    # --- surface texture ----------------------------------------------------
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_and(gray, gray, mask=mask)
    texture_energy = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    return {
        "dark_spot_ratio": round(dark_spot_ratio, 4),
        "mould_ratio": round(mould_ratio, 4),
        "mean_saturation": round(mean_saturation, 2),
        "texture_energy": round(texture_energy, 2),
    }


def cv_freshness_score(features: dict) -> float:
    """Convert raw CV features into a 0-100 freshness score."""
    score = 100.0

    # Decay spots are the strongest signal
    score -= min(features["dark_spot_ratio"] * 160, 60)

    # Visible mould is close to disqualifying
    score -= min(features["mould_ratio"] * 220, 45)

    # Dull colour means the item is drying out
    if features["mean_saturation"] < 90:
        score -= (90 - features["mean_saturation"]) * 0.30

    # Very rough surface = wrinkling / shrivelling
    if features["texture_energy"] > 600:
        score -= min((features["texture_energy"] - 600) / 45, 15)

    return float(np.clip(score, 0, 100))


# ----------------------------------------------------------------------------
# Signal 1: CNN fresh vs rotten
# ----------------------------------------------------------------------------
def build_model() -> tf.keras.Model:
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*config.IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False

    inputs = layers.Input(shape=(*config.IMG_SIZE, 3))
    x = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.15),
            layers.RandomContrast(0.2),
        ]
    )(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="freshness")(x)

    model = models.Model(inputs, outputs, name="freshness_detector")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def train(data_dir: str | Path, epochs: int = 15):
    data_dir = Path(data_dir)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir / "train",
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        label_mode="binary",
        shuffle=True,
        seed=config.SEED,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir / "val",
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        label_mode="binary",
        shuffle=False,
    )
    class_names = train_ds.class_names   # e.g. ['fresh', 'rotten']
    print(f"Classes: {class_names}")

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    model = build_model()
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc", mode="max", patience=4, restore_best_weights=True
            )
        ],
    )

    loss, acc, auc = model.evaluate(val_ds)
    print(f"\nValidation accuracy {acc * 100:.2f}% | AUC {auc:.3f}")

    model.save(config.FRESHNESS_MODEL_PATH)
    config.FRESHNESS_LABELS_PATH.write_text(json.dumps(class_names, indent=2))
    print(f"Saved model -> {config.FRESHNESS_MODEL_PATH}")
    return model


# ----------------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------------
class FreshnessDetector:
    """
    Fuses the CNN prediction with CV evidence.

    If no trained CNN is present the detector still works using the CV
    features alone, so you can demo the pipeline before training finishes.
    """

    def __init__(self, model_path: Path | None = None):
        model_path = Path(model_path or config.FRESHNESS_MODEL_PATH)
        self.model = None
        self.fresh_index = 0

        if model_path.exists():
            self.model = tf.keras.models.load_model(model_path)
            if config.FRESHNESS_LABELS_PATH.exists():
                labels = json.loads(config.FRESHNESS_LABELS_PATH.read_text())
                # sigmoid output 1.0 corresponds to labels[1]
                self.fresh_index = labels.index("fresh") if "fresh" in labels else 0
        else:
            print("[freshness] No CNN found - running on CV features only.")

    def _cnn_score(self, image_path: str | Path) -> float | None:
        if self.model is None:
            return None
        img = tf.keras.utils.load_img(image_path, target_size=config.IMG_SIZE)
        arr = np.expand_dims(tf.keras.utils.img_to_array(img), axis=0)
        p = float(self.model.predict(arr, verbose=0)[0][0])
        prob_fresh = p if self.fresh_index == 1 else 1.0 - p
        return prob_fresh * 100.0

    def predict(self, image_path: str | Path, food_name: str = "item") -> FreshnessResult:
        features = extract_cv_features(image_path)
        cv_score = cv_freshness_score(features)
        cnn_score = self._cnn_score(image_path)

        if cnn_score is None:
            final = cv_score
        else:
            final = CNN_WEIGHT * cnn_score + CV_WEIGHT * cv_score

        # Hard override: clearly visible mould is never "fresh"
        if features["mould_ratio"] > 0.18:
            final = min(final, 25.0)

        status, advice = _band_for(final)

        # Map the score onto remaining days, capped by the item's shelf life
        max_days = config.DEFAULT_SHELF_LIFE_DAYS.get(
            food_name, config.FALLBACK_SHELF_LIFE_DAYS
        )
        days_left = int(round((final / 100.0) * max_days))
        if status == "SPOILED":
            days_left = 0

        return FreshnessResult(
            food_name=food_name,
            score=round(final, 1),
            status=status,
            advice=advice,
            estimated_days_left=days_left,
            evidence={
                "cnn_score": None if cnn_score is None else round(cnn_score, 1),
                "cv_score": round(cv_score, 1),
                **features,
            },
        )

    def explain(self, result: FreshnessResult) -> str:
        """Plain-language reason, shown on the fridge display / app."""
        e = result.evidence
        reasons = []
        if e["dark_spot_ratio"] > 0.10:
            reasons.append(f"{e['dark_spot_ratio'] * 100:.0f}% of the surface is browning")
        if e["mould_ratio"] > 0.08:
            reasons.append("possible mould growth detected")
        if e["mean_saturation"] < 90:
            reasons.append("colour has faded")
        if e["texture_energy"] > 600:
            reasons.append("surface looks wrinkled")
        if not reasons:
            return "Colour, texture and surface all look normal."
        return "Detected: " + ", ".join(reasons) + "."


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ZeroWaste AI - freshness detection")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--data", default="data/freshness")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--predict", metavar="IMAGE")
    parser.add_argument("--food", default="item", help="food name, e.g. tomato")
    args = parser.parse_args()

    if args.train:
        train(args.data, epochs=args.epochs)
    elif args.predict:
        detector = FreshnessDetector()
        result = detector.predict(args.predict, args.food)
        print(f"\n{result}")
        print(f"Advice : {result.advice}")
        print(f"Reason : {detector.explain(result)}")
        print(f"Evidence: {result.evidence}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
