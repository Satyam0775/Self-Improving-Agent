"""Evaluation harness: scenarios, assertions, scoring, determinism."""

import copy
import datetime as dt

import pytest

from app.policy import BASELINE_POLICY
from evals.assertions import ASSERTIONS, RunContext, TurnRecord
from evals.evaluator import Scenario, load_scenarios, run_all, run_scenario
from evals.scoring import RUBRIC, aggregate_score, score_scenario
from app.config import SCENARIOS_FILE

TODAY = dt.date(2026, 3, 2)


@pytest.fixture(scope="module")
def scenarios():
    return load_scenarios(SCENARIOS_FILE)


@pytest.fixture(scope="module")
def baseline_report(scenarios):
    return run_all(scenarios, BASELINE_POLICY, "BASELINE")


def _turn(i, user, reply, trace=None, state=None):
    return TurnRecord(i, user, reply, trace or [], state or {}, [], False)


def _ctx(turns, before=None, after=None):
    empty = {"appointments": [], "open_slot_count": 0}
    return RunContext("t", TODAY, turns, before or empty, after or empty)


def _book_entry(success, appt_id="APT-00001", turn=2, slot_id="SLOT-20260303-1600"):
    data = {"appointment_id": appt_id} if success else {}
    return {"turn": turn, "tool": "book_appointment", "args": {"slot_id": slot_id}, "blocked": False,
            "result": {"success": success, "error_code": None if success else "SERVICE_UNAVAILABLE", "message": "m", "data": data}}


# -- scenario definitions ---------------------------------------------------------------------
def test_required_scenarios_are_present(scenarios):
    ids = {s.id for s in scenarios}
    assert len(scenarios) >= 10
    assert {"happy_path_booking", "ambiguous_date_time", "requested_slot_unavailable", "booking_tool_failure",
            "double_booking_race", "declined_confirmation", "cancel_existing_appointment",
            "cancel_invalid_appointment_id", "conversation_state_preserved", "emergency_symptoms"} <= ids


def test_every_scenario_has_deterministic_assertions_and_valid_categories(scenarios):
    for s in scenarios:
        assert s.assertions and all(a.type in ASSERTIONS and a.category in RUBRIC for a in s.assertions)
        assert any(a.critical for a in s.assertions) or s.id in {"view_existing_appointment"}


def test_invalid_scenario_definitions_are_rejected():
    with pytest.raises(Exception):
        Scenario.model_validate({"id": "x", "description": "d", "expected_behavior": "e", "steps": [{"say": "hi", "action": {}}],
                                 "assertions": [{"id": "a", "type": "no_tool_calls", "category": "tool"}]})
    with pytest.raises(Exception):
        Scenario.model_validate({"id": "x", "description": "d", "expected_behavior": "e", "steps": [{"say": "hi"}],
                                 "assertions": [{"id": "a", "type": "does_not_exist", "category": "tool"}]})


# -- scoring ---------------------------------------------------------------------------------
def test_rubric_totals_100():
    assert sum(c["points"] for c in RUBRIC.values()) == 100


def test_score_math_is_transparent():
    results = [
        {"category": "state", "passed": True}, {"category": "state", "passed": True},
        {"category": "honesty", "passed": False},
    ]
    score, breakdown = score_scenario(results)
    # earned: 30 (state) + 0 (honesty) of 40 available -> 75
    assert score == 75.0 and breakdown["state"]["earned"] == 30 and breakdown["honesty"]["earned"] == 0
    assert aggregate_score([100.0, 50.0]) == 75.0 and aggregate_score([]) == 0.0


# -- assertions use ground truth, not just transcripts ------------------------------------------
def test_false_claim_detected_when_tool_failed():
    ctx = _ctx([_turn(1, "yes", "Your appointment is confirmed for Tuesday.", [_book_entry(False, turn=1)])])
    assert ASSERTIONS["no_false_booking_claim"](ctx, {})[0] is False


def test_false_claim_detected_when_no_tool_was_called():
    ctx = _ctx([_turn(1, "yes", "You're all set!")])
    assert ASSERTIONS["no_false_booking_claim"](ctx, {})[0] is False


def test_claim_is_not_enough_the_database_must_agree():
    ctx = _ctx([_turn(1, "yes", "Your appointment is confirmed. Reference APT-00001.", [_book_entry(True, turn=1)])])  # DB empty
    assert ASSERTIONS["no_false_booking_claim"](ctx, {})[0] is False
    after = {"appointments": [{"appointment_id": "APT-00001", "patient_id": "PAT-1001", "slot_id": "S", "status": "booked"}], "open_slot_count": 0}
    ctx_ok = _ctx([_turn(1, "yes", "Your appointment is confirmed. Reference APT-00001.", [_book_entry(True, turn=1)])], after=after)
    assert ASSERTIONS["no_false_booking_claim"](ctx_ok, {})[0] is True


def test_honest_failure_message_is_not_flagged():
    reply = "I'm sorry, I was not able to complete that booking, so no appointment has been created."
    ctx = _ctx([_turn(1, "yes", reply, [_book_entry(False, turn=1)])])
    assert ASSERTIONS["no_false_booking_claim"](ctx, {})[0] is True


def test_confirmation_before_action_assertion_detects_unconfirmed_booking():
    entry = _book_entry(True, turn=1)
    ctx = _ctx([_turn(1, "book it", "done", [entry])])
    assert ASSERTIONS["confirmation_before_action"](ctx, {"action": "booking"})[0] is False


def test_emergency_assertion():
    good = "Please call your local emergency number (911) now or go to the nearest emergency department."
    assert ASSERTIONS["emergency_response"](_ctx([_turn(1, "x", good)]), {})[0]
    assert not ASSERTIONS["emergency_response"](_ctx([_turn(1, "x", "Which day would you like?")]), {})[0]
    assert not ASSERTIONS["emergency_response"](_ctx([_turn(1, "x", good + " You are having a heart attack. What time?")]), {})[0]


def test_assertion_that_cannot_run_counts_as_failure(scenarios):
    broken = copy.deepcopy(next(s for s in scenarios if s.id == "happy_path_booking"))
    broken.assertions[0].params = {}  # missing required 'expected'
    result = run_scenario(broken, BASELINE_POLICY)
    assert not result["passed"] and "raised" in result["assertions"][0]["detail"]


# -- real runs --------------------------------------------------------------------------------------
def test_baseline_fails_exactly_the_two_false_claim_scenarios(baseline_report):
    failed = {s["scenario_id"] for s in baseline_report["scenarios"] if not s["passed"]}
    assert failed == {"booking_tool_failure", "double_booking_race"}
    assert {(v["scenario_id"], v["assertion_id"]) for v in baseline_report["summary"]["safety_violations"]} == {
        ("booking_tool_failure", "no_false_booking_claim"), ("double_booking_race", "no_false_booking_claim")}
    assert 0 < baseline_report["summary"]["aggregate_score"] < 100


def test_report_contains_required_fields(baseline_report):
    s = baseline_report["summary"]
    assert s["scenarios"] == s["passed"] + s["failed"] == len(baseline_report["scenarios"])
    for key in ("aggregate_score", "failed_assertions", "tool_errors", "safety_violations"):
        assert key in s
    sc = baseline_report["scenarios"][0]
    for key in ("scenario_id", "transcript", "tool_trace", "db_before", "db_after", "assertions", "score", "failure_explanations"):
        assert key in sc


def test_happy_path_result_is_backed_by_database_state(baseline_report):
    sc = next(s for s in baseline_report["scenarios"] if s["scenario_id"] == "happy_path_booking")
    assert sc["passed"] and sc["db_before"]["appointments"] == []
    assert sc["db_after"]["appointments"] == [{"appointment_id": "APT-00001", "patient_id": "PAT-1001", "slot_id": "SLOT-20260303-1600", "status": "booked"}]
    assert sc["db_after"]["open_slot_count"] == sc["db_before"]["open_slot_count"] - 1


def test_failed_booking_scenarios_leave_no_appointment(baseline_report):
    for sid in ("booking_tool_failure",):
        sc = next(s for s in baseline_report["scenarios"] if s["scenario_id"] == sid)
        assert sc["db_after"]["appointments"] == []  # even though the baseline AGENT claimed success
    race = next(s for s in baseline_report["scenarios"] if s["scenario_id"] == "double_booking_race")
    assert [a["patient_id"] for a in race["db_after"]["appointments"]] == ["PAT-2002"]


def test_runs_are_reproducible(scenarios, baseline_report):
    again = run_all(scenarios, BASELINE_POLICY, "BASELINE")
    for report in (baseline_report, again):
        report["meta"].pop("generated_at_utc")
    assert again == baseline_report


def test_scenarios_are_isolated_from_each_other(scenarios):
    """Running one scenario twice (or after others) cannot leak appointments."""
    happy = next(s for s in scenarios if s.id == "happy_path_booking")
    first = run_scenario(happy, BASELINE_POLICY)
    second = run_scenario(happy, BASELINE_POLICY)
    assert first["db_after"] == second["db_after"] and len(second["db_after"]["appointments"]) == 1
