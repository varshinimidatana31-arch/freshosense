"""
ZeroWaste AI - pipeline
=======================

Ties the three modules into the flow described in the project deck:

    camera image
        -> Module 1: what food is this?
        -> Module 3: if it is raw produce, how fresh is it?
        -> Module 2: store it with an expiry date and alert as it nears

Usage:
    python main.py --scan tomato.jpg            # full pipeline on one image
    python main.py --scan-shelf data/shelf/     # a whole shelf of photos
    python main.py --daily                      # run today's alert check
    python main.py --demo                       # no images needed, seeds sample data
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import config
from alerts import Alert, AlertManager, ConsoleChannel
from expiry_monitor import ExpiryMonitor, InventoryDB


class ZeroWasteAI:
    def __init__(self):
        self.db = InventoryDB()
        self.alerts = AlertManager(channels=[ConsoleChannel()])
        self.monitor = ExpiryMonitor(self.db, self.alerts)
        self._identifier = None
        self._freshness = None

    # Models are loaded lazily so --daily and --demo work without TensorFlow
    @property
    def identifier(self):
        if self._identifier is None:
            from food_identification import FoodIdentifier
            self._identifier = FoodIdentifier()
        return self._identifier

    @property
    def freshness(self):
        if self._freshness is None:
            from freshness_detection import FreshnessDetector
            self._freshness = FreshnessDetector()
        return self._freshness

    # ------------------------------------------------------------------
    def scan_image(self, image_path: str | Path, quantity: float = 1, unit: str = "pcs"):
        """Run one captured image through all three modules."""
        print(f"\n--- Scanning {image_path} ---")

        # Module 1 -------------------------------------------------------
        prediction = self.identifier.predict(image_path)
        print(f"1. Identified : {prediction}")
        if not prediction.is_confident:
            print("   Low confidence - the app should ask the user to confirm.")
            print(f"   Alternatives: {prediction.top_k[1:]}")

        # Module 3 -------------------------------------------------------
        freshness_days = None
        score = None
        if prediction.is_raw_food:
            result = self.freshness.predict(image_path, prediction.label)
            score = result.score
            freshness_days = result.estimated_days_left
            print(f"2. Freshness  : {result.status} ({result.score:.0f}/100)")
            print(f"   {self.freshness.explain(result)}")

            if not result.is_safe_to_eat:
                self.alerts.dispatch(
                    Alert(
                        item_id=-1,
                        item_name=prediction.label,
                        severity="CRITICAL",
                        category="FRESHNESS",
                        title=f"Spoiled {prediction.label} detected",
                        message=(
                            f"The {prediction.label} placed in the fridge appears "
                            f"spoiled (freshness {result.score:.0f}/100). "
                            f"{result.advice}"
                        ),
                    ),
                    force=True,
                )
                return None
        else:
            print("2. Freshness  : skipped (packaged item)")

        # Module 2 -------------------------------------------------------
        item = self.db.add_item(
            prediction.label,
            quantity=quantity,
            unit=unit,
            freshness_days=freshness_days,
            freshness_score=score,
        )
        print(f"3. Inventory  : {item}")
        print(f"   Expiry set from: {item.source}")
        return item

    def scan_shelf(self, folder: str | Path):
        """Batch mode for the internal camera's periodic shelf sweep."""
        folder = Path(folder)
        images = sorted(
            p for p in folder.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
        if not images:
            print(f"No images found in {folder}")
            return
        print(f"Found {len(images)} image(s) in {folder}")
        for image in images:
            self.scan_image(image)

    # ------------------------------------------------------------------
    def daily_check(self):
        self.monitor.run_daily_check()

    def show_inventory(self):
        items = self.db.all_items()
        if not items:
            print("Fridge inventory is empty.")
            return
        print(f"\n{'':<4}{'ITEM':<15}{'QTY':<11}STATUS")
        print("-" * 52)
        for item in items:
            print(item)
        print("-" * 52)
        s = self.monitor.summary()
        print(f"Total {s['total']}   Expiring soon {s['expiring_soon']}   "
              f"Expired {s['expired']}")

    # ------------------------------------------------------------------
    def seed_demo_data(self):
        """Sample items with staged expiry dates, so alerts actually fire."""
        today = date.today()
        samples = [
            ("milk", 1, "L", today + timedelta(days=1)),
            ("spinach", 250, "g", today + timedelta(days=0)),
            ("tomato", 4, "pcs", today + timedelta(days=3)),
            ("chicken", 500, "g", today - timedelta(days=1)),
            ("yogurt", 2, "cups", today + timedelta(days=9)),
            ("apple", 6, "pcs", today + timedelta(days=21)),
        ]
        for name, qty, unit, expiry in samples:
            self.db.add_item(name, qty, unit, expiry_date=expiry)
        print(f"Seeded {len(samples)} demo items into {config.DB_PATH.name}")


def main():
    parser = argparse.ArgumentParser(description="ZeroWaste AI pipeline")
    parser.add_argument("--scan", metavar="IMAGE", help="run one image through all modules")
    parser.add_argument("--scan-shelf", metavar="FOLDER", help="batch scan a folder")
    parser.add_argument("--daily", action="store_true", help="run the expiry alert check")
    parser.add_argument("--inventory", action="store_true", help="print the inventory")
    parser.add_argument("--demo", action="store_true", help="seed demo data then alert")
    args = parser.parse_args()

    app = ZeroWasteAI()

    if args.scan:
        app.scan_image(args.scan)
        app.daily_check()
    elif args.scan_shelf:
        app.scan_shelf(args.scan_shelf)
        app.daily_check()
    elif args.demo:
        app.seed_demo_data()
        app.show_inventory()
        app.daily_check()
    elif args.inventory:
        app.show_inventory()
    elif args.daily:
        app.daily_check()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
