"""
MODULE 2 - Expiry Monitoring
=============================

Keeps the fridge inventory in SQLite, works out when each item expires, and
raises alert messages as the deadline approaches.

Expiry date comes from one of three sources, in priority order:

  1. A date the user typed in, or one OCR'd off the product label.
  2. The freshness module's estimate for raw produce with no printed date.
  3. The default refrigerated shelf-life table in config.py.

Alerts fire at the day thresholds in config.EXPIRY_ALERT_DAYS (3, 1, 0 by
default) and again once an item is past its date.

Usage:
    python expiry_monitor.py --add tomato --qty 4
    python expiry_monitor.py --list
    python expiry_monitor.py --check
"""

from __future__ import annotations

import argparse
import re
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import config
from alerts import Alert, AlertManager, ConsoleChannel


# ----------------------------------------------------------------------------
# Data model
# ----------------------------------------------------------------------------
@dataclass
class FoodItem:
    id: int
    name: str
    quantity: float
    unit: str
    added_on: date
    expiry_date: date
    freshness_score: float | None
    source: str          # how the expiry date was decided
    consumed: bool = False

    @property
    def days_left(self) -> int:
        return (self.expiry_date - date.today()).days

    @property
    def is_expired(self) -> bool:
        return self.days_left < 0

    def __str__(self) -> str:
        if self.is_expired:
            when = f"EXPIRED {abs(self.days_left)} day(s) ago"
        elif self.days_left == 0:
            when = "expires TODAY"
        else:
            when = f"{self.days_left} day(s) left"
        qty = f"{self.quantity:g} {self.unit}"
        return f"#{self.id:<3} {self.name:<14} {qty:<10} {when}"


# ----------------------------------------------------------------------------
# Expiry date helpers
# ----------------------------------------------------------------------------
DATE_PATTERNS = [
    (r"(\d{2})[/\-.](\d{2})[/\-.](\d{4})", "%d/%m/%Y"),
    (r"(\d{4})[/\-.](\d{2})[/\-.](\d{2})", "%Y/%m/%d"),
    (r"(\d{2})[/\-.](\d{2})[/\-.](\d{2})", "%d/%m/%y"),
]


def parse_printed_date(text: str) -> date | None:
    """
    Pull an expiry date out of OCR text taken from a product label.
    Handles the common 'EXP 12/08/2026', 'Best before 2026-08-12' forms.
    """
    if not text:
        return None
    cleaned = text.upper().replace("BEST BEFORE", "").replace("EXP", "")
    for pattern, fmt in DATE_PATTERNS:
        match = re.search(pattern, cleaned)
        if match:
            try:
                return datetime.strptime(match.group(0).replace("-", "/").replace(".", "/"),
                                         fmt.replace("-", "/")).date()
            except ValueError:
                continue
    return None


def estimate_expiry(food_name: str, freshness_days: int | None = None) -> tuple[date, str]:
    """Return (expiry_date, source_description)."""
    if freshness_days is not None:
        return date.today() + timedelta(days=freshness_days), "freshness_model"
    shelf_life = config.DEFAULT_SHELF_LIFE_DAYS.get(
        food_name, config.FALLBACK_SHELF_LIFE_DAYS
    )
    return date.today() + timedelta(days=shelf_life), "shelf_life_table"


# ----------------------------------------------------------------------------
# Inventory store
# ----------------------------------------------------------------------------
class InventoryDB:
    def __init__(self, db_path=None):
        self.db_path = str(db_path or config.DB_PATH)
        self._init_schema()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS inventory (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    name            TEXT NOT NULL,
                    quantity        REAL NOT NULL DEFAULT 1,
                    unit            TEXT NOT NULL DEFAULT 'pcs',
                    added_on        TEXT NOT NULL,
                    expiry_date     TEXT NOT NULL,
                    freshness_score REAL,
                    source          TEXT NOT NULL,
                    consumed        INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_expiry ON inventory(expiry_date)")

    # -- writes ------------------------------------------------------------
    def add_item(
        self,
        name: str,
        quantity: float = 1,
        unit: str = "pcs",
        expiry_date: date | None = None,
        freshness_days: int | None = None,
        freshness_score: float | None = None,
    ) -> FoodItem:
        if expiry_date is not None:
            source = "user_or_ocr"
        else:
            expiry_date, source = estimate_expiry(name, freshness_days)

        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO inventory
                   (name, quantity, unit, added_on, expiry_date, freshness_score, source)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    name,
                    quantity,
                    unit,
                    date.today().isoformat(),
                    expiry_date.isoformat(),
                    freshness_score,
                    source,
                ),
            )
            item_id = cur.lastrowid

        return FoodItem(
            id=item_id,
            name=name,
            quantity=quantity,
            unit=unit,
            added_on=date.today(),
            expiry_date=expiry_date,
            freshness_score=freshness_score,
            source=source,
        )

    def mark_consumed(self, item_id: int):
        with self._connect() as conn:
            conn.execute("UPDATE inventory SET consumed=1 WHERE id=?", (item_id,))

    def remove_item(self, item_id: int):
        with self._connect() as conn:
            conn.execute("DELETE FROM inventory WHERE id=?", (item_id,))

    def update_freshness(self, item_id: int, score: float, days_left: int):
        """Called after the internal camera re-scans an item."""
        new_expiry = date.today() + timedelta(days=days_left)
        with self._connect() as conn:
            conn.execute(
                "UPDATE inventory SET freshness_score=?, expiry_date=?, source=? WHERE id=?",
                (score, new_expiry.isoformat(), "freshness_model", item_id),
            )

    # -- reads -------------------------------------------------------------
    @staticmethod
    def _row_to_item(row: sqlite3.Row) -> FoodItem:
        return FoodItem(
            id=row["id"],
            name=row["name"],
            quantity=row["quantity"],
            unit=row["unit"],
            added_on=date.fromisoformat(row["added_on"]),
            expiry_date=date.fromisoformat(row["expiry_date"]),
            freshness_score=row["freshness_score"],
            source=row["source"],
            consumed=bool(row["consumed"]),
        )

    def all_items(self, include_consumed: bool = False) -> list[FoodItem]:
        query = "SELECT * FROM inventory"
        if not include_consumed:
            query += " WHERE consumed=0"
        query += " ORDER BY expiry_date ASC"
        with self._connect() as conn:
            return [self._row_to_item(r) for r in conn.execute(query)]

    def expiring_within(self, days: int) -> list[FoodItem]:
        cutoff = (date.today() + timedelta(days=days)).isoformat()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM inventory WHERE consumed=0 AND expiry_date<=? "
                "ORDER BY expiry_date ASC",
                (cutoff,),
            )
            return [self._row_to_item(r) for r in rows]


# ----------------------------------------------------------------------------
# The monitor itself
# ----------------------------------------------------------------------------
class ExpiryMonitor:
    """Scans the inventory and produces alert messages."""

    def __init__(self, db: InventoryDB | None = None, alert_manager: AlertManager | None = None):
        self.db = db or InventoryDB()
        self.alerts = alert_manager or AlertManager(channels=[ConsoleChannel()])

    # -- message building --------------------------------------------------
    @staticmethod
    def _build_alert(item: FoodItem) -> Alert | None:
        days = item.days_left

        if days < 0:
            return Alert(
                item_id=item.id,
                item_name=item.name,
                severity="CRITICAL",
                category="EXPIRY",
                title=f"{item.name.title()} has expired",
                message=(
                    f"{item.quantity:g} {item.unit} of {item.name} expired "
                    f"{abs(days)} day(s) ago. Please remove it from the fridge."
                ),
            )

        if days == 0:
            return Alert(
                item_id=item.id,
                item_name=item.name,
                severity="CRITICAL",
                category="EXPIRY",
                title=f"{item.name.title()} expires today",
                message=(
                    f"{item.quantity:g} {item.unit} of {item.name} expires today. "
                    f"Use it now or check the app for a recipe that uses it."
                ),
            )

        if days in config.EXPIRY_ALERT_DAYS:
            severity = "WARNING" if days == 1 else "INFO"
            return Alert(
                item_id=item.id,
                item_name=item.name,
                severity=severity,
                category="EXPIRY",
                title=f"{item.name.title()} expires in {days} day(s)",
                message=(
                    f"{item.quantity:g} {item.unit} of {item.name} will expire on "
                    f"{item.expiry_date.strftime('%d %b %Y')}. Plan to use it soon."
                ),
            )
        return None

    # -- public API --------------------------------------------------------
    def scan(self) -> list[Alert]:
        """Return every alert that is due right now, without sending."""
        alerts = []
        for item in self.db.all_items():
            alert = self._build_alert(item)
            if alert:
                alerts.append(alert)
        return alerts

    def run_daily_check(self) -> int:
        """Scan + dispatch. Schedule this once a day (cron / APScheduler)."""
        alerts = self.scan()
        if not alerts:
            print("No expiry alerts today. Everything in the fridge is fine.")
            return 0

        print(f"\n===== EXPIRY ALERTS  ({date.today():%d %b %Y}) =====")
        sent = self.alerts.dispatch_all(alerts)
        print(f"===== {sent} alert(s) delivered =====\n")
        return sent

    def summary(self) -> dict:
        """Counts for the fridge display's inventory overview tiles."""
        items = self.db.all_items()
        return {
            "total": len(items),
            "expiring_soon": sum(1 for i in items if 0 <= i.days_left <= 3),
            "expired": sum(1 for i in items if i.is_expired),
        }


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ZeroWaste AI - expiry monitoring")
    parser.add_argument("--add", metavar="FOOD", help="add an item to the inventory")
    parser.add_argument("--qty", type=float, default=1)
    parser.add_argument("--unit", default="pcs")
    parser.add_argument("--expiry", help="expiry date as YYYY-MM-DD (optional)")
    parser.add_argument("--list", action="store_true", help="show the inventory")
    parser.add_argument("--check", action="store_true", help="run the daily alert check")
    parser.add_argument("--consume", type=int, metavar="ID", help="mark an item used")
    args = parser.parse_args()

    monitor = ExpiryMonitor()

    if args.add:
        expiry = date.fromisoformat(args.expiry) if args.expiry else None
        item = monitor.db.add_item(args.add, args.qty, args.unit, expiry_date=expiry)
        print(f"Added: {item}   (expiry source: {item.source})")
    elif args.consume:
        monitor.db.mark_consumed(args.consume)
        print(f"Item #{args.consume} marked as consumed.")
    elif args.list:
        items = monitor.db.all_items()
        if not items:
            print("Fridge inventory is empty.")
        else:
            print(f"\n{'':<4}{'ITEM':<15}{'QTY':<11}STATUS")
            print("-" * 52)
            for item in items:
                print(item)
            print("-" * 52)
            print(monitor.summary())
    elif args.check:
        monitor.run_daily_check()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
