"""
Alert delivery for ZeroWaste AI.

The expiry monitor decides *what* to warn about; this file decides *how* the
warning reaches the user. Channels are pluggable - the console channel always
works, email works once SMTP credentials are set in config.py, and the push
channel is a stub you can wire to Firebase Cloud Messaging (the deck lists
Firebase as the project's notification backend).

De-duplication is handled here so the same item does not nag the user twice
on the same day.
"""

from __future__ import annotations

import smtplib
import sqlite3
from dataclasses import dataclass
from datetime import date
from email.message import EmailMessage

import config


SEVERITY_ICON = {
    "INFO": "[i]",
    "WARNING": "[!]",
    "CRITICAL": "[X]",
}


@dataclass
class Alert:
    item_id: int
    item_name: str
    severity: str        # INFO | WARNING | CRITICAL
    title: str
    message: str
    category: str        # EXPIRY | FRESHNESS

    def format(self) -> str:
        icon = SEVERITY_ICON.get(self.severity, "[-]")
        return f"{icon} {self.title}\n    {self.message}"


# ----------------------------------------------------------------------------
# Channels
# ----------------------------------------------------------------------------
class ConsoleChannel:
    name = "console"

    def send(self, alert: Alert) -> bool:
        print(alert.format())
        return True


class EmailChannel:
    name = "email"

    def send(self, alert: Alert) -> bool:
        if not (config.SMTP_USER and config.SMTP_PASSWORD and config.ALERT_RECIPIENT):
            return False
        msg = EmailMessage()
        msg["Subject"] = f"[ZeroWaste AI] {alert.title}"
        msg["From"] = config.SMTP_USER
        msg["To"] = config.ALERT_RECIPIENT
        msg.set_content(alert.message)
        try:
            with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT, timeout=15) as server:
                server.starttls()
                server.login(config.SMTP_USER, config.SMTP_PASSWORD)
                server.send_message(msg)
            return True
        except Exception as exc:                      # noqa: BLE001
            print(f"[alerts] email failed: {exc}")
            return False


class PushChannel:
    """
    Firebase Cloud Messaging stub.

    To enable it: pip install firebase-admin, drop your service-account JSON
    in the project folder, and replace the body of send() with the two
    commented lines.
    """

    name = "push"

    def __init__(self, device_token: str | None = None):
        self.device_token = device_token

    def send(self, alert: Alert) -> bool:
        if not self.device_token:
            return False
        # from firebase_admin import messaging
        # messaging.send(messaging.Message(
        #     notification=messaging.Notification(alert.title, alert.message),
        #     token=self.device_token))
        print(f"[push -> {self.device_token[:8]}...] {alert.title}")
        return True


# ----------------------------------------------------------------------------
# Manager
# ----------------------------------------------------------------------------
class AlertManager:
    """Fans an alert out to every registered channel, once per item per day."""

    def __init__(self, channels=None, db_path=None):
        self.channels = channels or [ConsoleChannel()]
        self.db_path = str(db_path or config.DB_PATH)
        self._init_log()

    def _init_log(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS alert_log (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_id     INTEGER NOT NULL,
                    category    TEXT NOT NULL,
                    severity    TEXT NOT NULL,
                    message     TEXT NOT NULL,
                    sent_on     TEXT NOT NULL,
                    UNIQUE(item_id, category, severity, sent_on)
                )
                """
            )

    def _already_sent_today(self, alert: Alert) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """SELECT 1 FROM alert_log
                   WHERE item_id=? AND category=? AND severity=? AND sent_on=?""",
                (alert.item_id, alert.category, alert.severity, date.today().isoformat()),
            ).fetchone()
        return row is not None

    def _record(self, alert: Alert):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR IGNORE INTO alert_log
                   (item_id, category, severity, message, sent_on)
                   VALUES (?,?,?,?,?)""",
                (
                    alert.item_id,
                    alert.category,
                    alert.severity,
                    alert.message,
                    date.today().isoformat(),
                ),
            )

    def dispatch(self, alert: Alert, force: bool = False) -> bool:
        if not force and self._already_sent_today(alert):
            return False
        delivered = any(channel.send(alert) for channel in self.channels)
        if delivered:
            self._record(alert)
        return delivered

    def dispatch_all(self, alerts: list[Alert]) -> int:
        return sum(1 for a in alerts if self.dispatch(a))

    def history(self, limit: int = 20) -> list[tuple]:
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute(
                "SELECT sent_on, severity, message FROM alert_log "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
