"""SQLite persistence for the simulated clinic schedule and appointments.

Double-booking protection is enforced by the database itself:

* a partial UNIQUE index allows at most one ``booked`` appointment per slot;
* every write runs inside ``BEGIN IMMEDIATE`` so competing writers are serialised.

A fresh connection is opened per operation, which keeps the code simple and safe
to use from several threads or processes (and avoids file locks on Windows).
"""

from __future__ import annotations

import datetime as dt
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

DEFAULT_SLOT_TIMES = ("09:00", "10:00", "11:00", "14:00", "16:00")
DEFAULT_SCHEDULE_DAYS = 14
DEFAULT_CONSULTATION_TYPE = "general consultation"

SCHEMA = """
CREATE TABLE IF NOT EXISTS slots (
    slot_id           TEXT PRIMARY KEY,
    slot_date         TEXT NOT NULL,
    slot_time         TEXT NOT NULL,
    consultation_type TEXT NOT NULL DEFAULT 'general consultation',
    UNIQUE (slot_date, slot_time)
);
CREATE TABLE IF NOT EXISTS appointments (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id   TEXT NOT NULL,
    slot_id      TEXT NOT NULL REFERENCES slots(slot_id),
    status       TEXT NOT NULL CHECK (status IN ('booked', 'cancelled')),
    created_at   TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    cancelled_at TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_one_active_booking_per_slot
    ON appointments(slot_id) WHERE status = 'booked';
"""


class DatabaseError(Exception):
    """Base class for expected, domain-level database errors."""


class SlotNotFound(DatabaseError):
    pass


class SlotUnavailable(DatabaseError):
    pass


class AppointmentNotFound(DatabaseError):
    pass


class AlreadyCancelled(DatabaseError):
    pass


def appointment_ref(number: int) -> str:
    return f"APT-{number:05d}"


def appointment_number(ref: str) -> int:
    return int(ref.split("-", 1)[1])


def make_slot_id(day: dt.date, hhmm: str) -> str:
    return f"SLOT-{day:%Y%m%d}-{hhmm.replace(':', '')}"


def _slot_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "slot_id": row["slot_id"],
        "date": row["slot_date"],
        "time": row["slot_time"],
        "consultation_type": row["consultation_type"],
    }


class Database:
    def __init__(self, path: str | Path):
        self.path = str(path)

    # -- connections ----------------------------------------------------
    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            else:
                conn.execute("COMMIT")

    # -- setup ----------------------------------------------------------
    def init_schema(self) -> None:
        with self.connection() as conn:
            conn.executescript(SCHEMA)

    def seed_schedule(
        self,
        today: dt.date,
        days: int = DEFAULT_SCHEDULE_DAYS,
        times: tuple[str, ...] = DEFAULT_SLOT_TIMES,
    ) -> int:
        """Create weekday slots for the ``days`` days after ``today`` (idempotent)."""
        created = 0
        with self.transaction() as conn:
            for offset in range(1, days + 1):
                day = today + dt.timedelta(days=offset)
                if day.weekday() >= 5:  # no weekend clinic
                    continue
                for hhmm in times:
                    cur = conn.execute(
                        "INSERT OR IGNORE INTO slots (slot_id, slot_date, slot_time, consultation_type) "
                        "VALUES (?, ?, ?, ?)",
                        (make_slot_id(day, hhmm), day.isoformat(), hhmm, DEFAULT_CONSULTATION_TYPE),
                    )
                    created += cur.rowcount
        return created

    # -- reads ----------------------------------------------------------
    def open_slots(self, date_iso: str, time_preference: str = "any") -> list[dict[str, Any]]:
        clause = ""
        if time_preference == "morning":
            clause = "AND s.slot_time < '12:00'"
        elif time_preference == "afternoon":
            clause = "AND s.slot_time >= '12:00'"
        sql = f"""
            SELECT s.* FROM slots s
            WHERE s.slot_date = ? {clause}
              AND NOT EXISTS (SELECT 1 FROM appointments a
                              WHERE a.slot_id = s.slot_id AND a.status = 'booked')
            ORDER BY s.slot_time
        """
        with self.connection() as conn:
            return [_slot_dict(r) for r in conn.execute(sql, (date_iso,))]

    def get_slot(self, slot_id: str) -> Optional[dict[str, Any]]:
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM slots WHERE slot_id = ?", (slot_id,)).fetchone()
            if row is None:
                return None
            booked = conn.execute(
                "SELECT 1 FROM appointments WHERE slot_id = ? AND status = 'booked'", (slot_id,)
            ).fetchone()
            data = _slot_dict(row)
            data["available"] = booked is None
            return data

    _APPT_SQL = """
        SELECT a.id, a.patient_id, a.slot_id, a.status,
               s.slot_date, s.slot_time, s.consultation_type
        FROM appointments a JOIN slots s ON s.slot_id = a.slot_id
    """

    @staticmethod
    def _appt_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "appointment_id": appointment_ref(row["id"]),
            "patient_id": row["patient_id"],
            "slot_id": row["slot_id"],
            "date": row["slot_date"],
            "time": row["slot_time"],
            "consultation_type": row["consultation_type"],
            "status": row["status"],
        }

    def get_appointment(self, ref: str) -> dict[str, Any]:
        with self.connection() as conn:
            row = conn.execute(self._APPT_SQL + " WHERE a.id = ?", (appointment_number(ref),)).fetchone()
        if row is None:
            raise AppointmentNotFound(ref)
        return self._appt_dict(row)

    # -- writes ---------------------------------------------------------
    def book(self, patient_id: str, slot_id: str) -> dict[str, Any]:
        """Book ``slot_id``; raises SlotNotFound / SlotUnavailable."""
        with self.transaction() as conn:
            if conn.execute("SELECT 1 FROM slots WHERE slot_id = ?", (slot_id,)).fetchone() is None:
                raise SlotNotFound(slot_id)
            try:
                cur = conn.execute(
                    "INSERT INTO appointments (patient_id, slot_id, status) VALUES (?, ?, 'booked')",
                    (patient_id, slot_id),
                )
            except sqlite3.IntegrityError:
                raise SlotUnavailable(slot_id) from None
            new_id = cur.lastrowid
        return self.get_appointment(appointment_ref(new_id))

    def cancel(self, ref: str) -> dict[str, Any]:
        """Cancel a booked appointment; raises AppointmentNotFound / AlreadyCancelled."""
        with self.transaction() as conn:
            row = conn.execute("SELECT status FROM appointments WHERE id = ?", (appointment_number(ref),)).fetchone()
            if row is None:
                raise AppointmentNotFound(ref)
            if row["status"] != "booked":
                raise AlreadyCancelled(ref)
            conn.execute(
                "UPDATE appointments SET status = 'cancelled', cancelled_at = CURRENT_TIMESTAMP "
                "WHERE id = ? AND status = 'booked'",
                (appointment_number(ref),),
            )
        return self.get_appointment(ref)

    # -- evaluation support ---------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        """Deterministic view of the database for evaluation (no timestamps)."""
        with self.connection() as conn:
            appts = [
                {
                    "appointment_id": appointment_ref(r["id"]),
                    "patient_id": r["patient_id"],
                    "slot_id": r["slot_id"],
                    "status": r["status"],
                }
                for r in conn.execute("SELECT id, patient_id, slot_id, status FROM appointments ORDER BY id")
            ]
            open_count = conn.execute(
                "SELECT COUNT(*) FROM slots s WHERE NOT EXISTS "
                "(SELECT 1 FROM appointments a WHERE a.slot_id = s.slot_id AND a.status = 'booked')"
            ).fetchone()[0]
        return {"appointments": appts, "open_slot_count": open_count}
