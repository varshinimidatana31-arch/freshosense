# ZeroWaste AI — Python Modules

Implementation of three modules from *Integrated Intelligent Food Management
System for Smart Refrigerators*:

| File | Module | What it does |
|---|---|---|
| `food_identification.py` | Food Identification | CNN that names the food item in a camera image |
| `expiry_monitor.py` | Expiry Monitoring | Inventory database + alert messages before expiry |
| `freshness_detection.py` | Raw Food Detection | Fresh/spoiled prediction with a 0–100 freshness score |
| `alerts.py` | — | Alert delivery (console, email, Firebase push) |
| `main.py` | — | Pipeline that connects all three |
| `config.py` | — | All constants: classes, shelf-life table, thresholds |

## Setup

```bash
pip install -r requirements.txt
```

## Try it without any training

The expiry module needs no model, so you can demo it immediately:

```bash
python main.py --demo
```

That seeds six sample items and prints the alerts they trigger.

## Datasets you need

Both CNNs use folder-per-class structure. Two Kaggle datasets that match:

* **Food identification** — *Fruits and Vegetables Image Recognition Dataset*, or
  *Grocery Store Dataset*. Arrange as:

```
data/food/train/apple/...    data/food/val/apple/...
data/food/train/tomato/...   data/food/val/tomato/...
```

* **Freshness** — *Fresh and Rotten Fruits/Vegetables*. Arrange as:

```
data/freshness/train/fresh/...    data/freshness/val/fresh/...
data/freshness/train/rotten/...   data/freshness/val/rotten/...
```

After collecting the food dataset, set `FOOD_CLASSES` in `config.py` to your
folder names and add a shelf-life value for each in `DEFAULT_SHELF_LIFE_DAYS`.

## Training

```bash
python food_identification.py --train --data data/food --epochs 15
python freshness_detection.py  --train --data data/freshness --epochs 15
```

Both use MobileNetV2 transfer learning — ImageNet weights frozen first, then
the top 30 layers fine-tuned at a low learning rate. On a Colab T4 this takes
roughly 15–25 minutes per model. The food model trains in two stages because
fine-tuning from the start destroys the pretrained features.

## Running each module

```bash
# Module 1
python food_identification.py --predict test_images/tomato.jpg

# Module 3
python freshness_detection.py --predict test_images/tomato.jpg --food tomato

# Module 2
python expiry_monitor.py --add milk --qty 1 --unit L
python expiry_monitor.py --list
python expiry_monitor.py --check

# All three together
python main.py --scan test_images/tomato.jpg
```

## How the modules connect

```
camera image
   ↓
Module 1  identify → "tomato", 94% confident
   ↓  (raw produce? yes)
Module 3  freshness → score 72/100, ~5 days left
   ↓
Module 2  store with expiry = today + 5 days
   ↓
daily scan → alert at 3 days, 1 day, day-of, and after expiry
```

Expiry dates come from three sources in priority order: a date the user typed
or OCR'd off the label, the freshness model's estimate, then the shelf-life
table. Packaged goods skip Module 3 entirely.

## Design notes worth mentioning in your viva

**Why MobileNetV2, not a CNN from scratch.** A fridge dataset is a few thousand
images at most; training from scratch overfits. MobileNetV2 is also ~14 MB and
converts to TFLite, so it can run on the fridge's board instead of the cloud —
matching the on-device requirement in your architecture.

**Why the freshness module fuses CNN with OpenCV.** A CNN alone gives a number
with no explanation. The classical features — brown-spot ratio, colour
saturation, texture roughness, mould-tone ratio — let the app tell the user
*why* something was flagged ("38% of the surface is browning"), which is what
makes the alert trustworthy. It also means the module still works before the
CNN is trained. The fusion weights are `CNN_WEIGHT = 0.65` / `CV_WEIGHT = 0.35`
in `freshness_detection.py`; tune them on your validation set.

**Why alerts are de-duplicated in the database.** Without the `alert_log`
table, running the daily check twice sends the same warning twice. The
UNIQUE constraint on `(item_id, category, severity, sent_on)` makes the check
idempotent, so you can schedule it freely.

## Scheduling the daily check

```python
from apscheduler.schedulers.blocking import BlockingScheduler
from expiry_monitor import ExpiryMonitor

scheduler = BlockingScheduler()
scheduler.add_job(ExpiryMonitor().run_daily_check, "cron", hour=8)
scheduler.start()
```

Or as a cron entry:

```
0 8 * * *  cd /path/to/zerowaste_ai && python main.py --daily
```

## Enabling real notifications

In `config.py`, set `SMTP_USER`, `SMTP_PASSWORD` (a Gmail app password, not
your account password) and `ALERT_RECIPIENT`, then register the channel:

```python
from alerts import AlertManager, ConsoleChannel, EmailChannel
manager = AlertManager(channels=[ConsoleChannel(), EmailChannel()])
```

For the Flutter app, `PushChannel` in `alerts.py` has the Firebase Cloud
Messaging call commented out — uncomment it and add your service-account
credentials.

## Not included

The recipe recommendation, smart shopping list, and OCR barcode scanning
modules from the deck aren't here. `parse_printed_date()` in `expiry_monitor.py`
is the hook where OCR output plugs in — pass it the text pytesseract returns
from a label photo.
