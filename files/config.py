"""
ZeroWaste AI - central configuration.

Every module reads its constants from here so you only change values in one place.
"""

from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
DB_PATH = BASE_DIR / "fridge_inventory.db"

MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

FOOD_MODEL_PATH = MODEL_DIR / "food_identifier.keras"
FOOD_LABELS_PATH = MODEL_DIR / "food_labels.json"
FRESHNESS_MODEL_PATH = MODEL_DIR / "freshness_detector.keras"
FRESHNESS_LABELS_PATH = MODEL_DIR / "freshness_labels.json"

# ----------------------------------------------------------------------------
# Image settings (MobileNetV2 expects 224x224)
# ----------------------------------------------------------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# ----------------------------------------------------------------------------
# Food classes the identification model is trained on.
# Replace with your own dataset's folder names.
# ----------------------------------------------------------------------------
FOOD_CLASSES = [
    "apple", "banana", "bell_pepper", "bread", "broccoli", "butter",
    "carrot", "cheese", "chicken", "cucumber", "egg", "fish",
    "grapes", "lettuce", "milk", "mango", "onion", "orange",
    "potato", "spinach", "strawberry", "tomato", "yogurt",
]

# Which of those are raw/perishable produce that the freshness module
# should also be run on.
RAW_FOOD_CLASSES = {
    "apple", "banana", "bell_pepper", "broccoli", "carrot", "chicken",
    "cucumber", "fish", "grapes", "lettuce", "mango", "onion",
    "orange", "potato", "spinach", "strawberry", "tomato",
}

# ----------------------------------------------------------------------------
# Default refrigerated shelf life in days.
# Used when a product has no printed expiry date (loose fruit, vegetables).
# ----------------------------------------------------------------------------
DEFAULT_SHELF_LIFE_DAYS = {
    "apple": 30, "banana": 7, "bell_pepper": 12, "bread": 7,
    "broccoli": 7, "butter": 60, "carrot": 21, "cheese": 21,
    "chicken": 2, "cucumber": 7, "egg": 28, "fish": 2,
    "grapes": 10, "lettuce": 7, "milk": 7, "mango": 6,
    "onion": 30, "orange": 21, "potato": 60, "spinach": 5,
    "strawberry": 5, "tomato": 7, "yogurt": 14,
}
FALLBACK_SHELF_LIFE_DAYS = 7

# ----------------------------------------------------------------------------
# Alerting
# ----------------------------------------------------------------------------
# Send a reminder when an item has this many days left.
EXPIRY_ALERT_DAYS = [3, 1, 0]

# Freshness score bands (0-100, higher = fresher)
FRESHNESS_BANDS = [
    (80, "FRESH", "Good to eat."),
    (55, "USE_SOON", "Still edible - use within a day or two."),
    (30, "DEGRADING", "Quality dropping fast. Cook it today."),
    (0, "SPOILED", "Do not consume. Discard this item."),
]

# Confidence below which the system asks the user to confirm the food name
MIN_IDENTIFICATION_CONFIDENCE = 0.60

# Email channel (fill these in, or leave blank to disable email alerts)
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USER = ""          # e.g. "zerowaste.team7@gmail.com"
SMTP_PASSWORD = ""      # app password, never your real password
ALERT_RECIPIENT = ""    # user's email
