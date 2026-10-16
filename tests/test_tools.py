"""Deterministic tool tests: no LLM, no network."""

import sqlite3
import threading

from app.database import Database
from app.tools import AppointmentTools


def test_availability_lists_real_open_slots(tools, today):
    res = tools.check_availability("2026-03-03", "any")
    assert res.success
    assert [s["time"] for s in res.data["slots"]] == ["09:00", "10:00", "11:00", "14:00", "16:00"]


def test_availability_time_preference_filters(tools):
    morning = tools.check_availability("2026-03-03", "morning").data["slots"]
    afternoon = tools.check_availability("2026-03-03", "afternoon").data["slots"]
    assert [s["time"] for s in morning] == ["09:00", "10:00", "11:00"]
    assert [s["time"] for s in afternoon] == ["14:00", "16:00"]


def test_availability_weekend_and_out_of_range_are_empty(tools):
    assert tools.check_availability("2026-03-07").data["slots"] == []  # Saturday
    assert tools.check_availability("2027-01-01").data["slots"] == []  # outside schedule window


def test_availability_rejects_bad_dates(tools):
    for bad in ["tomorrow", "03/03/2026", "2026-02-30", ""]:
        res = tools.check_availability(bad)
        assert not res.success and res.error_code == "INVALID_INPUT"


def test_successful_booking_is_persisted_and_removes_availability(tools, db, slot):
    res = tools.book_appointment("PAT-1001", slot(1, "16:00"), True)
    assert res.success and res.data["appointment_id"] == "APT-00001"
    assert db.get_appointment("APT-00001")["status"] == "booked"
    remaining = [s["time"] for s in tools.check_availability("2026-03-03").data["slots"]]
    assert "16:00" not in remaining


def test_booking_requires_explicit_confirmation(tools, db, slot):
    res = tools.book_appointment("PAT-1001", slot(1, "16:00"), False)
    assert not res.success and res.error_code == "CONFIRMATION_REQUIRED"
    assert db.snapshot()["appointments"] == []
    # a truthy string is not a confirmation either (strict boolean)
    res = tools.book_appointment("PAT-1001", slot(1, "16:00"), "yes")
    assert not res.success and res.error_code == "INVALID_INPUT"
    assert db.snapshot()["appointments"] == []


def test_unavailable_and_nonexistent_slots(tools, db, slot):
    assert tools.book_appointment("PAT-1001", slot(1, "16:00"), True).success
    second = tools.book_appointment("PAT-2002", slot(1, "16:00"), True)
    assert not second.success and second.error_code == "SLOT_UNAVAILABLE"
    missing = tools.book_appointment("PAT-1001", slot(1, "15:00"), True)
    assert not missing.success and missing.error_code == "SLOT_NOT_FOUND"
    assert len(db.snapshot()["appointments"]) == 1


def test_double_booking_protection_under_concurrency(db, slot):
    results = []

    def attempt(n):
        t = AppointmentTools(Database(db.path))
        results.append(t.book_appointment(f"PAT-{3000 + n}", slot(2, "10:00"), True))

    threads = [threading.Thread(target=attempt, args=(n,)) for n in range(10)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sum(r.success for r in results) == 1
    assert all(r.error_code == "SLOT_UNAVAILABLE" for r in results if not r.success)
    assert len([a for a in db.snapshot()["appointments"] if a["status"] == "booked"]) == 1


def test_database_itself_rejects_duplicate_active_bookings(db, slot):
    """The constraint holds even if application code is bypassed entirely."""
    AppointmentTools(db).book_appointment("PAT-1001", slot(1, "09:00"), True)
    conn = sqlite3.connect(db.path)
    try:
        try:
            conn.execute("INSERT INTO appointments (patient_id, slot_id, status) VALUES ('PAT-9999', ?, 'booked')", (slot(1, "09:00"),))
            raised = False
        except sqlite3.IntegrityError:
            raised = True
    finally:
        conn.close()
    assert raised


def test_get_appointment_and_invalid_ids(tools, slot):
    tools.book_appointment("PAT-1001", slot(1, "11:00"), True)
    ok = tools.get_appointment("APT-00001")
    assert ok.success and ok.data["time"] == "11:00" and ok.data["status"] == "booked"
    assert tools.get_appointment("APT-99999").error_code == "APPOINTMENT_NOT_FOUND"
    for bad in ["banana", "APT-1", "apt-00001", "", "APT-0000X"]:
        assert tools.get_appointment(bad).error_code == "INVALID_INPUT"


def test_cancellation_flow_and_slot_is_released(tools, db, slot):
    tools.book_appointment("PAT-1001", slot(1, "16:00"), True)
    assert tools.cancel_appointment("APT-00001", False).error_code == "CONFIRMATION_REQUIRED"
    assert db.get_appointment("APT-00001")["status"] == "booked"
    res = tools.cancel_appointment("APT-00001", True)
    assert res.success and db.get_appointment("APT-00001")["status"] == "cancelled"
    assert tools.cancel_appointment("APT-00001", True).error_code == "ALREADY_CANCELLED"
    again = tools.book_appointment("PAT-2002", slot(1, "16:00"), True)  # slot can be rebooked
    assert again.success


def test_invalid_cancellation_does_not_touch_other_appointments(tools, db, slot):
    tools.book_appointment("PAT-1001", slot(1, "16:00"), True)
    before = db.snapshot()
    assert tools.cancel_appointment("APT-99999", True).error_code == "APPOINTMENT_NOT_FOUND"
    assert tools.cancel_appointment("not-an-id", True).error_code == "INVALID_INPUT"
    assert db.snapshot() == before


def test_injected_tool_failure_returns_error_and_writes_nothing(db, slot):
    failing = AppointmentTools(db, fail_tools={"book_appointment": "SERVICE_UNAVAILABLE"})
    res = failing.book_appointment("PAT-1001", slot(1, "16:00"), True)
    assert not res.success and res.error_code == "SERVICE_UNAVAILABLE"
    assert db.snapshot()["appointments"] == []


def test_validation_errors_do_not_echo_input(tools):
    res = tools.book_appointment("my-real-name-John-Smith", "SLOT-20260303-1600", True)
    assert res.error_code == "INVALID_INPUT"
    assert "John" not in res.message
