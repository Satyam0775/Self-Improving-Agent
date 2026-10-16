"""Deterministic appointment tools.

These are plain Python functions that return structured ``ToolResult`` objects.
They never call an LLM, so they can be tested (and evaluated) offline. The outcome
of a booking or cancellation is decided here, from the database, never from LLM text.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
from typing import Any, Optional

from pydantic import ValidationError

from .database import (
    AlreadyCancelled,
    AppointmentNotFound,
    Database,
    SlotNotFound,
    SlotUnavailable,
)
from .schemas import (
    BookInput,
    CancelInput,
    CheckAvailabilityInput,
    GetAppointmentInput,
    ToolResult,
)
from .utils import format_date, slot_label


def _fail(code: str, message: str, **data: Any) -> ToolResult:
    return ToolResult(success=False, error_code=code, message=message, data=data)


def _ok(message: str, **data: Any) -> ToolResult:
    return ToolResult(success=True, message=message, data=data)


def _validate(model: Any, **kwargs: Any) -> tuple[Optional[Any], Optional[ToolResult]]:
    """Validate inputs; the error message never echoes the offending value."""
    try:
        return model(**kwargs), None
    except ValidationError as exc:
        parts = [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()]
        return None, _fail("INVALID_INPUT", "Invalid input - " + "; ".join(parts))
    except TypeError as exc:  # unexpected keyword etc.
        return None, _fail("INVALID_INPUT", f"Invalid input - {exc}")


class AppointmentTools:
    """The four required tools plus a read-only ``describe_slot`` helper.

    ``fail_tools`` maps tool name -> error code and simulates an outage of that
    tool *without touching the database*. It is used by the evaluation harness.
    """

    def __init__(self, db: Database, fail_tools: Optional[dict[str, str]] = None):
        self.db = db
        self.fail_tools = dict(fail_tools or {})

    def _injected_failure(self, tool: str) -> Optional[ToolResult]:
        code = self.fail_tools.get(tool)
        if code:
            return _fail(code, "The scheduling service is temporarily unavailable. No changes were made.")
        return None

    # -- required tools -------------------------------------------------
    def check_availability(self, date: str, time_preference: str = "any") -> ToolResult:
        if (failure := self._injected_failure("check_availability")) is not None:
            return failure
        args, err = _validate(CheckAvailabilityInput, date=date, time_preference=time_preference)
        if err:
            return err
        try:
            slots = self.db.open_slots(args.date, args.time_preference)
        except sqlite3.Error:
            return _fail("INTERNAL_ERROR", "The schedule could not be read.")
        for s in slots:
            s["label"] = slot_label(s["date"], s["time"])
        day = dt.date.fromisoformat(args.date)
        date_label = format_date(day)  # authoritative weekday for this date; the model must quote it
        if slots:
            msg = f"{len(slots)} open slot(s) on {date_label} ({args.time_preference})."
        elif day.weekday() >= 5:
            msg = f"{date_label} is a weekend and the clinic is closed. No open slots."
        else:
            msg = f"No open slots on {date_label} for preference '{args.time_preference}'."
        return _ok(
            msg,
            date=args.date,
            weekday=day.strftime("%A"),
            date_label=date_label,
            time_preference=args.time_preference,
            slots=slots,
        )

    def describe_slot(self, slot_id: str) -> ToolResult:
        """Helper: is this slot real and still open? (read-only)"""
        if (failure := self._injected_failure("describe_slot")) is not None:
            return failure
        args, err = _validate(BookInput, patient_id="PAT-0000", slot_id=slot_id, confirmation=False)
        if err:
            return err
        slot = self.db.get_slot(args.slot_id)
        if slot is None:
            return _fail("SLOT_NOT_FOUND", "That time slot does not exist.", slot_id=args.slot_id)
        slot["label"] = slot_label(slot["date"], slot["time"])
        if not slot["available"]:
            return _fail("SLOT_UNAVAILABLE", "That time slot is already taken.", **slot)
        return _ok("Slot is open.", **slot)

    def book_appointment(self, patient_id: str, slot_id: str, confirmation: bool) -> ToolResult:
        if (failure := self._injected_failure("book_appointment")) is not None:
            return failure
        args, err = _validate(BookInput, patient_id=patient_id, slot_id=slot_id, confirmation=confirmation)
        if err:
            return err
        if args.confirmation is not True:
            return _fail("CONFIRMATION_REQUIRED", "Booking requires explicit patient confirmation.")
        try:
            appt = self.db.book(args.patient_id, args.slot_id)
        except SlotNotFound:
            return _fail("SLOT_NOT_FOUND", "That time slot does not exist.", slot_id=args.slot_id)
        except SlotUnavailable:
            return _fail("SLOT_UNAVAILABLE", "That time slot is no longer available.", slot_id=args.slot_id)
        except sqlite3.Error:
            return _fail("INTERNAL_ERROR", "The booking could not be saved. No appointment was created.")
        appt["label"] = slot_label(appt["date"], appt["time"])
        return _ok(f"Appointment {appt['appointment_id']} booked.", **appt)

    def get_appointment(self, appointment_id: str) -> ToolResult:
        if (failure := self._injected_failure("get_appointment")) is not None:
            return failure
        args, err = _validate(GetAppointmentInput, appointment_id=appointment_id)
        if err:
            return err
        try:
            appt = self.db.get_appointment(args.appointment_id)
        except AppointmentNotFound:
            return _fail("APPOINTMENT_NOT_FOUND", "No appointment exists with that reference.")
        except sqlite3.Error:
            return _fail("INTERNAL_ERROR", "The appointment could not be read.")
        appt["label"] = slot_label(appt["date"], appt["time"])
        return _ok(f"Appointment {appt['appointment_id']} is {appt['status']}.", **appt)

    def cancel_appointment(self, appointment_id: str, confirmation: bool) -> ToolResult:
        if (failure := self._injected_failure("cancel_appointment")) is not None:
            return failure
        args, err = _validate(CancelInput, appointment_id=appointment_id, confirmation=confirmation)
        if err:
            return err
        if args.confirmation is not True:
            return _fail("CONFIRMATION_REQUIRED", "Cancellation requires explicit patient confirmation.")
        try:
            appt = self.db.cancel(args.appointment_id)
        except AppointmentNotFound:
            return _fail("APPOINTMENT_NOT_FOUND", "No appointment exists with that reference.")
        except AlreadyCancelled:
            return _fail("ALREADY_CANCELLED", "That appointment was already cancelled.")
        except sqlite3.Error:
            return _fail("INTERNAL_ERROR", "The cancellation could not be saved. Nothing was changed.")
        appt["label"] = slot_label(appt["date"], appt["time"])
        return _ok(f"Appointment {appt['appointment_id']} cancelled.", **appt)

    # -- verification (used by guardrails; bypasses fault injection) ----
    def verify_appointment(self, appointment_id: str) -> Optional[dict[str, Any]]:
        """Ground-truth lookup straight from the database (None if missing)."""
        try:
            return self.db.get_appointment(appointment_id)
        except (AppointmentNotFound, ValueError, IndexError, sqlite3.Error):
            return None


# Model-facing tool specifications (used to build live-model function declarations).
# ``patient_id`` is deliberately absent: the executor injects the session's
# synthetic patient ID so the model cannot book on behalf of someone else.
MODEL_TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "check_availability",
        "description": (
            "List open appointment slots for a date. Always use this; never guess availability. "
            "The result's date_label has the correct weekday for the date; quote that, never your own."
        ),
        "properties": {
            "date": {"type": "STRING", "description": "ISO date YYYY-MM-DD taken from the calendar in your instructions."},
            "time_preference": {"type": "STRING", "enum": ["morning", "afternoon", "any"]},
        },
        "required": ["date"],
    },
    {
        "name": "propose_booking",
        "description": (
            "Record that you are about to ask the patient to confirm ONE specific slot. "
            "Call this, then ask the patient to confirm in plain words. Do not book in the same turn."
        ),
        "properties": {"slot_id": {"type": "STRING", "description": "slot_id from check_availability."}},
        "required": ["slot_id"],
    },
    {
        "name": "book_appointment",
        "description": (
            "Book the proposed slot. Only call after the patient explicitly said yes to that exact slot "
            "in their latest message. Report success only if the result has success=true."
        ),
        "properties": {
            "slot_id": {"type": "STRING", "description": "The slot the patient confirmed."},
            "confirmation": {"type": "BOOLEAN", "description": "True only if the patient explicitly confirmed."},
        },
        "required": ["slot_id", "confirmation"],
    },
    {
        "name": "get_appointment",
        "description": "Look up an appointment by reference (format APT-00001).",
        "properties": {"appointment_id": {"type": "STRING", "description": "Reference like APT-00001."}},
        "required": ["appointment_id"],
    },
    {
        "name": "propose_cancellation",
        "description": "Record that you are about to ask the patient to confirm cancelling ONE appointment.",
        "properties": {"appointment_id": {"type": "STRING", "description": "Reference like APT-00001."}},
        "required": ["appointment_id"],
    },
    {
        "name": "cancel_appointment",
        "description": (
            "Cancel the proposed appointment. Only call after the patient explicitly said yes in their "
            "latest message. Report success only if the result has success=true."
        ),
        "properties": {
            "appointment_id": {"type": "STRING", "description": "Reference like APT-00001."},
            "confirmation": {"type": "BOOLEAN", "description": "True only if the patient explicitly confirmed."},
        },
        "required": ["appointment_id", "confirmation"],
    },
]
