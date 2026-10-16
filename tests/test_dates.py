"""Date / weekday / availability consistency (regression for 'Monday 18 October' when it is a Sunday).

All offline: fixed 'today' of Friday 2026-10-16, fake Groq client, temporary database.
"""

import datetime as dt
import json

import pytest

from app.agent import AppointmentAgent
from app.database import Database, make_slot_id
from app.groq_backend import GroqBackend
from app.policy import BASELINE_POLICY
from app.prompts import build_calendar, build_system_prompt
from app.state import ConversationState
from app.tools import AppointmentTools
from app.utils import correct_weekdays, slot_label
from tests.test_groq_backend import FakeClient, call, calling, saying

LIVE_TODAY = dt.date(2026, 10, 16)  # Friday, the day of the live incident


@pytest.fixture
def live_db(tmp_path):
    database = Database(tmp_path / "live.db")
    database.init_schema()
    database.seed_schedule(LIVE_TODAY)
    return database


def test_fixture_dates_are_what_the_incident_assumed():
    assert LIVE_TODAY.strftime("%A") == "Friday"
    assert dt.date(2026, 10, 18).strftime("%A") == "Sunday"  # not Monday
    assert dt.date(2026, 10, 19).strftime("%A") == "Monday"


# -- calendar given to the model -------------------------------------------------------------
def test_prompt_calendar_pairs_every_date_with_its_real_weekday():
    lines = build_calendar(LIVE_TODAY).splitlines()
    assert len(lines) == 15
    for line in lines:
        iso, weekday = line.split()[:2]
        assert dt.date.fromisoformat(iso).strftime("%A") == weekday
    assert "2026-10-18 Sunday (CLOSED weekend)" in lines and "2026-10-19 Monday (open)" in lines
    assert lines[0].endswith("<- today") and lines[1].endswith("<- tomorrow")


def test_system_prompt_contains_calendar_and_no_self_calculation_instruction():
    prompt = build_system_prompt(BASELINE_POLICY, ConversationState(patient_id="PAT-1001"), LIVE_TODAY)
    assert "2026-10-18 Sunday (CLOSED weekend)" in prompt and "2026-10-19 Monday (open)" in prompt
    assert "Never\ncalculate weekdays yourself" in prompt and "yourself" in prompt


# -- tool results are the authority for weekday and availability -----------------------------
def test_sunday_has_no_slots_and_tool_reports_the_real_weekday(live_db):
    res = AppointmentTools(live_db).check_availability("2026-10-18", "any")
    assert res.success and res.data["slots"] == []
    assert res.data["weekday"] == "Sunday" and res.data["date_label"] == "Sunday 18 October 2026"
    assert "Sunday 18 October 2026" in res.message and "weekend" in res.message and "Monday" not in res.message


def test_monday_19_october_has_all_slots_and_correct_label(live_db):
    res = AppointmentTools(live_db).check_availability("2026-10-19", "any")
    assert res.data["weekday"] == "Monday" and res.data["date_label"] == "Monday 19 October 2026"
    assert [s["time"] for s in res.data["slots"]] == ["09:00", "10:00", "11:00", "14:00", "16:00"]
    assert all(s["label"].startswith("Monday 19 October 2026") for s in res.data["slots"])


def test_every_seeded_slot_label_has_the_correct_weekday(live_db):
    slots = [s for offset in range(1, 15) for s in live_db.open_slots((LIVE_TODAY + dt.timedelta(days=offset)).isoformat())]
    assert slots and all(dt.date.fromisoformat(s["date"]).weekday() < 5 for s in slots)  # no weekend slots
    for s in slots:
        assert slot_label(s["date"], s["time"]).split()[0] == dt.date.fromisoformat(s["date"]).strftime("%A")


def test_booked_monday_slot_is_removed_and_booking_label_is_correct(live_db):
    tools = AppointmentTools(live_db)
    booked = tools.book_appointment("PAT-1001", make_slot_id(dt.date(2026, 10, 19), "16:00"), True)
    assert booked.success and booked.data["label"] == "Monday 19 October 2026 at 4:00 PM"
    remaining = [s["time"] for s in tools.check_availability("2026-10-19").data["slots"]]
    assert "16:00" not in remaining and len(remaining) == 4


# -- reply-level weekday correction ------------------------------------------------------------
@pytest.mark.parametrize(
    "text,expected",
    [
        ("Monday 18 October has no open slots.", "Sunday 18 October has no open slots."),
        ("Monday, 18 October 2026 is closed.", "Sunday, 18 October 2026 is closed."),
        ("I can offer Tuesday, October 20 at 4 PM.", "I can offer Tuesday, October 20 at 4 PM."),  # already right
        ("Monday 19 October 2026 at 4:00 PM is booked.", "Monday 19 October 2026 at 4:00 PM is booked."),
        ("Friday 19th October works.", "Monday 19th October works."),
        ("See you on Monday.", "See you on Monday."),  # weekday without a date is left alone
        ("Monday 31 November is invalid.", "Monday 31 November is invalid."),  # impossible date is left alone
    ],
)
def test_wrong_weekday_for_a_date_is_corrected(text, expected):
    fixed, _ = correct_weekdays(text, LIVE_TODAY)
    assert fixed == expected


def test_correction_is_reported_and_correct_text_is_untouched():
    _, fixes = correct_weekdays("Monday 18 October is full.", LIVE_TODAY)
    assert fixes == ["Monday 18 October -> Sunday"]
    assert correct_weekdays("Monday 19 October is open.", LIVE_TODAY)[1] == []


# -- end to end with a (fake) live model ----------------------------------------------------------
def _agent(db, client, tools=None):
    return AppointmentAgent(tools or AppointmentTools(db), BASELINE_POLICY, GroqBackend(client, "test-model"), LIVE_TODAY)


def test_model_saying_monday_18_october_is_corrected_and_availability_comes_from_the_tool(live_db):
    client = FakeClient(
        calling(call("check_availability", date="2026-10-18", time_preference="any")),
        saying("Sorry, Monday 18 October 2026 has no open slots."),
    )
    res = _agent(live_db, client).handle_message("Anything on Monday?")
    assert "Sunday 18 October 2026" in res.reply and "Monday 18 October" not in res.reply
    assert any(g["rule_id"] == "CORE-WEEKDAY-CONSISTENCY" for g in res.guard_interventions)
    tool_msg = next(m for m in client.requests[1]["messages"] if m.get("role") == "tool")
    payload = json.loads(tool_msg["content"])["result"]
    assert payload["data"]["slots"] == [] and payload["data"]["date_label"] == "Sunday 18 October 2026"
    assert live_db.snapshot()["appointments"] == []


def test_full_live_style_booking_on_monday_19_october_keeps_confirmation_gate(live_db):
    slot = make_slot_id(dt.date(2026, 10, 19), "16:00")
    client = FakeClient(
        calling(call("check_availability", date="2026-10-19", time_preference="afternoon")),
        calling(call("propose_booking", slot_id=slot)),
        saying("I can book Monday 19 October 2026 at 4:00 PM. Shall I confirm?"),
        calling(call("book_appointment", slot_id=slot, confirmation=True)),
        saying("Your appointment is confirmed for Monday 19 October 2026 at 4:00 PM. Reference APT-00001."),
    )
    agent = _agent(live_db, client)
    first = agent.handle_message("Monday afternoon please")
    assert live_db.snapshot()["appointments"] == [] and first.guard_interventions == []
    second = agent.handle_message("Yes")
    assert "Monday 19 October 2026" in second.reply and "APT-00001" in second.reply
    assert [a["status"] for a in live_db.snapshot()["appointments"]] == ["booked"]
