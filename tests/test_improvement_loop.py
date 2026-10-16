"""Improvement loop: proposal validation, isolation, acceptance gate, regression rejection, idempotency."""

import json
from types import SimpleNamespace

import pytest

from app.config import SCENARIOS_FILE
from app.errors import PolicyError
from app.guardrails import GUARD_REGISTRY, temporary_guard
from app.policy import BASELINE_POLICY, load_active_policy, load_records
from app.schemas import ImprovementRecord, Target
from evals.evaluator import load_scenarios, run_all, run_scenario
from evals.improvement_loop import (
    NEGATIVE_CONTROL_GUARD,
    GroqProposer,
    _overblock_guard,
    analyze_failures,
    apply_proposal,
    decide,
    negative_control_proposal,
    parse_proposal,
    propose_from_failures,
    run_cycle,
    run_demo,
)


@pytest.fixture(scope="module")
def scenarios():
    return load_scenarios(SCENARIOS_FILE)


@pytest.fixture(scope="module")
def baseline(scenarios):
    return run_all(scenarios, BASELINE_POLICY, "BASELINE")


@pytest.fixture(scope="module")
def proposal(baseline):
    return propose_from_failures(analyze_failures(baseline))[0]


def _raw(**overrides):
    raw = {
        "id": "IMP-050", "type": "guardrail", "source_scenario": "booking_tool_failure",
        "targets": [{"scenario_id": "booking_tool_failure", "assertion_id": "no_false_booking_claim"}],
        "failure_description": "The agent claimed success after a tool error.",
        "rule": "Never claim a booking succeeded unless the tool succeeded and the database confirms it.",
        "rule_id": "GR-TEST-RULE", "enforcement": "verify_action_claims", "params": {"action": "booking"}, "status": "proposed",
    }
    raw.update(overrides)
    return raw


# -- failure analysis and proposal generation --------------------------------------------------
def test_failure_analysis_finds_the_real_failures_with_evidence(baseline):
    failures = analyze_failures(baseline)
    root = [f for f in failures if f["type"] == "no_false_booking_claim"]
    assert {f["scenario_id"] for f in root} == {"booking_tool_failure", "double_booking_race"}
    ev = root[0]["evidence"]
    assert "confirmed" in ev["agent_reply"].lower() and ev["tool_calls"][0]["success"] is False


def test_template_proposal_is_structured_valid_and_targets_the_failures(proposal):
    assert proposal.id == "IMP-001" and proposal.status == "proposed" and proposal.type == "guardrail"
    assert proposal.source_scenario == "booking_tool_failure"
    assert {(t.scenario_id, t.assertion_id) for t in proposal.targets} == {
        ("booking_tool_failure", "no_false_booking_claim"), ("double_booking_race", "no_false_booking_claim")}
    ImprovementRecord.model_validate(proposal.model_dump(mode="json"))  # round-trips through the strict schema


def test_no_failures_means_no_proposals(scenarios, proposal):
    good = run_all(scenarios, BASELINE_POLICY.with_rule(proposal.to_policy_rule()), "OK")
    assert analyze_failures(good) == []
    assert propose_from_failures(analyze_failures(good)) == []


# -- proposal validation ----------------------------------------------------------------------------
def test_valid_raw_proposal_parses():
    assert parse_proposal(_raw()).rule_id == "GR-TEST-RULE"


@pytest.mark.parametrize(
    "override",
    [
        {"enforcement": "exec_python"},  # not an allowed guard
        {"params": {"action": "everything"}},  # value outside allowed set
        {"params": {"shell": "rm -rf"}},  # unknown param
        {"rule": "Skip patient confirmation before booking to save time."},  # weakens a core invariant
        {"rule": "Ignore emergency symptoms when the patient is in a hurry."},
        {"rule": "Run eval(subprocess) to patch agent.py when a booking fails."},  # code-like content
        {"status": "accepted"},  # cannot self-accept
        {"rule_id": "lowercase-id"},  # schema pattern
        {"targets": []},
        {"extra_field": "x"},  # extra keys forbidden
    ],
)
def test_unsafe_or_malformed_proposals_are_rejected(override):
    with pytest.raises(PolicyError):
        parse_proposal(_raw(**override))


def test_duplicate_active_rule_is_rejected():
    with pytest.raises(PolicyError):
        parse_proposal(_raw(rule_id="GR-TEST-RULE"), active_rule_ids={"GR-TEST-RULE"})


def test_apply_proposal_returns_new_policy_and_leaves_baseline_untouched(proposal):
    fingerprint = BASELINE_POLICY.fingerprint()
    candidate = apply_proposal(BASELINE_POLICY, proposal)
    assert BASELINE_POLICY.fingerprint() == fingerprint and BASELINE_POLICY.rules == []
    assert candidate is not BASELINE_POLICY and [r.id for r in candidate.rules] == ["GR-VERIFY-BOOKING-CLAIM"]
    assert candidate.core_invariants_intact()


# -- acceptance --------------------------------------------------------------------------------------------
def test_real_improvement_is_accepted_with_measured_gain(scenarios, baseline, proposal):
    cycle = run_cycle(proposal, BASELINE_POLICY, baseline, scenarios)
    assert cycle.record.status == "accepted"
    assert cycle.record.baseline_score < cycle.record.candidate_score == 100.0
    assert cycle.record.regressions == []
    assert all(c["passed"] for c in cycle.decision["checks"].values())
    fixed = cycle.decision["comparison"]["assertions_fixed"]
    assert "booking_tool_failure::no_false_booking_claim" in fixed and "double_booking_race::no_false_booking_claim" in fixed
    assert any(e.get("kind") == "candidate_result" and e["passed"] for e in cycle.record.evidence)


def test_candidate_run_really_uses_the_policy(scenarios, proposal):
    """Same scenario, same model: only the policy differs, and the behaviour changes."""
    failing = next(s for s in scenarios if s.id == "booking_tool_failure")
    before = run_scenario(failing, BASELINE_POLICY)
    after = run_scenario(failing, apply_proposal(BASELINE_POLICY, proposal))
    assert not before["passed"] and after["passed"]
    assert after["guard_interventions"] and not before["guard_interventions"]
    assert after["db_after"] == before["db_after"]  # DB identical: only the honesty of the reply changed


# -- rejection paths -----------------------------------------------------------------------------------------
def test_regression_causes_rejection_and_reason_is_preserved(scenarios, baseline, proposal):
    with temporary_guard(NEGATIVE_CONTROL_GUARD, _overblock_guard):
        cycle = run_cycle(negative_control_proposal(proposal), BASELINE_POLICY, baseline, scenarios)
    assert cycle.record.status == "rejected"
    assert "happy_path_booking::real_reference_quoted" in cycle.record.regressions
    assert "no_assertion_regressions" in cycle.record.decision_reason and cycle.record.decision_reason.startswith("Rejected")
    assert NEGATIVE_CONTROL_GUARD not in GUARD_REGISTRY  # temporary guard cleaned up


def test_aggregate_score_alone_cannot_buy_acceptance(scenarios, baseline, proposal):
    """The negative control scores HIGHER than baseline but is still rejected for regressions."""
    with temporary_guard(NEGATIVE_CONTROL_GUARD, _overblock_guard):
        cycle = run_cycle(negative_control_proposal(proposal), BASELINE_POLICY, baseline, scenarios)
    assert cycle.record.candidate_score > cycle.record.baseline_score
    assert cycle.record.status == "rejected"


def test_unresolved_target_causes_rejection(scenarios, baseline, proposal):
    wrong = proposal.model_copy(update={"params": {"action": "cancellation"}, "rule_id": "GR-WRONG-GUARD"})
    cycle = run_cycle(wrong, BASELINE_POLICY, baseline, scenarios)
    assert cycle.record.status == "rejected"
    assert "target_failures_fixed" in cycle.record.decision_reason


def test_invalid_proposal_is_rejected_before_any_evaluation(scenarios, baseline, proposal):
    bad = proposal.model_copy(update={"enforcement": "does_not_exist"})
    cycle = run_cycle(bad, BASELINE_POLICY, baseline, scenarios)
    assert cycle.record.status == "rejected" and cycle.candidate_report is None
    assert "before evaluation" in cycle.record.decision_reason


def test_gate_rejects_missing_core_invariant_and_critical_failure(scenarios, baseline, proposal):
    cycle = run_cycle(proposal, BASELINE_POLICY, baseline, scenarios)
    weakened = cycle.candidate_policy.model_copy(update={"core_invariants": ["emergency_escalation"]})
    d = decide(baseline, cycle.candidate_report, proposal, BASELINE_POLICY, weakened, BASELINE_POLICY.fingerprint())
    assert not d["accepted"] and not d["checks"]["core_invariants_intact"]["passed"]
    # candidate that still has a critical failure (baseline report reused as "candidate")
    d2 = decide(baseline, baseline, proposal, BASELINE_POLICY, cycle.candidate_policy, BASELINE_POLICY.fingerprint())
    assert not d2["accepted"] and not d2["checks"]["no_critical_safety_failures"]["passed"] and not d2["checks"]["target_failures_fixed"]["passed"]


# -- whole demo: files, idempotency, no contamination -------------------------------------------------------------
def test_full_demo_writes_reports_and_records(tmp_path):
    out = run_demo(SCENARIOS_FILE, tmp_path / "reports", tmp_path / "imp", verbose=False)
    assert out["exit_code"] == 0
    for name in ("baseline_report.json", "candidate_report.json", "comparison_report.json"):
        assert (tmp_path / "reports" / name).exists()
    comparison = json.loads((tmp_path / "reports" / "comparison_report.json").read_text())
    imp = comparison["improvements"][0]
    assert imp["record"]["status"] == "accepted" and imp["decision"]["comparison"]["score_delta"] > 0
    assert comparison["negative_control"]["record"]["status"] == "rejected"
    accepted = load_records(tmp_path / "imp" / "accepted.json")
    proposals = load_records(tmp_path / "imp" / "proposals.json")
    assert [r.id for r in accepted] == ["IMP-001"] and {r.id: r.status for r in proposals} == {"IMP-001": "accepted", "IMP-900": "rejected"}


def test_accepted_policy_is_loaded_and_changes_live_agent_behaviour(tmp_path, scenarios):
    run_demo(SCENARIOS_FILE, tmp_path / "r", tmp_path / "imp", verbose=False)
    active = load_active_policy(tmp_path / "imp")
    assert [r.id for r in active.rules] == ["GR-VERIFY-BOOKING-CLAIM"]
    failing = next(s for s in scenarios if s.id == "booking_tool_failure")
    assert run_scenario(failing, active)["passed"] and not run_scenario(failing, BASELINE_POLICY)["passed"]


def test_repeated_runs_are_idempotent_and_baseline_is_never_contaminated(tmp_path):
    def snapshot(base):
        def clean(p):
            data = json.loads(p.read_text())
            data.get("meta", {}).pop("generated_at_utc", None)
            return data
        return {p.name: clean(p) for p in sorted(base.rglob("*.json"))}

    first = run_demo(SCENARIOS_FILE, tmp_path / "reports", tmp_path / "imp", verbose=False)
    snap1 = snapshot(tmp_path)
    second = run_demo(SCENARIOS_FILE, tmp_path / "reports", tmp_path / "imp", verbose=False)  # accepted.json already exists
    snap2 = snapshot(tmp_path)
    assert second["exit_code"] == 0
    assert snap1 == snap2  # identical files: no duplicate records, same scores
    assert first["baseline_report"]["summary"]["aggregate_score"] == second["baseline_report"]["summary"]["aggregate_score"] < 100
    assert second["baseline_report"]["summary"]["failed"] == 2  # baseline still fails: not contaminated by accepted.json


def test_rejected_improvement_is_not_activated_and_stale_acceptance_is_removed(tmp_path, monkeypatch):
    """If a later run rejects IMP-001, accepted.json must no longer contain it and the exit code must flag it."""
    import dataclasses

    from evals import improvement_loop as loop

    run_demo(SCENARIOS_FILE, tmp_path / "r", tmp_path / "imp", verbose=False)
    assert [r.id for r in load_records(tmp_path / "imp" / "accepted.json")] == ["IMP-001"]

    # Make the template propose a guard that cannot fix the target failures (cancellation guard for booking claims).
    broken = dataclasses.replace(loop.TEMPLATES["no_false_booking_claim"], params={"action": "cancellation"})
    monkeypatch.setitem(loop.TEMPLATES, "no_false_booking_claim", broken)
    out = run_demo(SCENARIOS_FILE, tmp_path / "r", tmp_path / "imp", verbose=False)

    assert out["exit_code"] == 2
    assert out["cycles"][0].record.status == "rejected"
    assert "target_failures_fixed" in out["cycles"][0].record.decision_reason
    assert load_records(tmp_path / "imp" / "accepted.json") == []
    assert load_active_policy(tmp_path / "imp").rules == []
    stored = {r.id: r for r in load_records(tmp_path / "imp" / "proposals.json")}
    assert stored["IMP-001"].status == "rejected" and stored["IMP-001"].decision_reason.startswith("Rejected")


# -- optional Groq proposer (fake client; no network) ------------------------------------------------------------------
class _FakeClient:
    def __init__(self, text):
        message = SimpleNamespace(content=text)
        choice = SimpleNamespace(message=message)
        completion = SimpleNamespace(create=lambda **kw: SimpleNamespace(choices=[choice]))
        self.chat = SimpleNamespace(completions=completion)


def _failures(baseline):
    return [f for f in analyze_failures(baseline) if f["type"] == "no_false_booking_claim"]


def test_groq_proposal_is_schema_validated_and_usable(baseline):
    draft = {"failure_description": "Agent claimed booking success after tool error.", "rule": "Only state a booking succeeded when the tool result and database confirm it.",
             "rule_id": "GR-GROQ-VERIFY", "enforcement": "verify_action_claims", "params": {"action": "booking"}}
    rec = GroqProposer(_FakeClient("```json\n" + json.dumps(draft) + "\n```"), "m").propose(_failures(baseline))
    assert rec.id == "IMP-101" and rec.status == "proposed" and len(rec.targets) == 2  # targets set by code, not by the LLM


@pytest.mark.parametrize(
    "text",
    [
        "not json at all",
        "[1, 2, 3]",
        json.dumps({"failure_description": "x" * 20, "rule": "Skip confirmation to book faster for patients.", "rule_id": "GR-BAD", "enforcement": "verify_action_claims", "params": {"action": "booking"}}),
        json.dumps({"failure_description": "x" * 20, "rule": "Patch the code to always say yes.", "rule_id": "GR-BAD", "enforcement": "edit_python_file", "params": {}}),
    ],
)
def test_bad_groq_proposals_are_refused(baseline, text):
    with pytest.raises(PolicyError):
        GroqProposer(_FakeClient(text), "m").propose(_failures(baseline))
