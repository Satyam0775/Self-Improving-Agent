"""Scenario runner: executes scenarios against an agent + policy and builds a report.

Every scenario runs on its own fresh temporary SQLite database, with a fixed
"today", so results are reproducible and no state leaks between scenarios or runs.
"""

from __future__ import annotations

import datetime as dt
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.agent import AppointmentAgent, Backend
from app.database import Database, make_slot_id
from app.schemas import Policy
from app.simulated_model import SimulatedModel
from app.tools import AppointmentTools

from .assertions import ASSERTIONS, RunContext, TurnRecord
from .scoring import RUBRIC, aggregate_score, score_scenario

EVAL_TODAY = dt.date(2026, 3, 2)  # a Monday; fixed so every run is identical
EVAL_PATIENT_ID = "PAT-1001"

BackendFactory = Callable[[dt.date], Backend]


def simulated_backend_factory(today: dt.date) -> Backend:
    return SimulatedModel(today, optimistic_booking_claims=True)


# ----------------------------------------------------------------------------
# Scenario schema
# ----------------------------------------------------------------------------
class BookedSlot(BaseModel):
    model_config = ConfigDict(extra="forbid")
    day_offset: int
    time: str
    patient_id: str


class Setup(BaseModel):
    model_config = ConfigDict(extra="forbid")
    booked: list[BookedSlot] = Field(default_factory=list)
    fail_tools: dict[str, str] = Field(default_factory=dict)


class Step(BaseModel):
    model_config = ConfigDict(extra="forbid")
    say: Optional[str] = None
    action: Optional[dict[str, Any]] = None

    @model_validator(mode="after")
    def _exactly_one(self) -> "Step":
        if (self.say is None) == (self.action is None):
            raise ValueError("a step needs exactly one of 'say' or 'action'")
        return self


class AssertionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    type: str
    category: str
    critical: bool = False
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _collect_params(cls, data: Any) -> Any:
        if isinstance(data, dict):
            known = {"id", "type", "category", "critical", "params"}
            data = dict(data)
            params = dict(data.get("params", {}))
            for key in [k for k in data if k not in known]:
                params[key] = data.pop(key)
            data["params"] = params
        return data

    @model_validator(mode="after")
    def _known(self) -> "AssertionSpec":
        if self.type not in ASSERTIONS:
            raise ValueError(f"unknown assertion type '{self.type}'")
        if self.category not in RUBRIC:
            raise ValueError(f"unknown rubric category '{self.category}'")
        return self


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    description: str
    expected_behavior: str
    setup: Setup = Field(default_factory=Setup)
    steps: list[Step]
    assertions: list[AssertionSpec]

    @model_validator(mode="after")
    def _unique_ids(self) -> "Scenario":
        ids = [a.id for a in self.assertions]
        if len(ids) != len(set(ids)):
            raise ValueError(f"scenario {self.id}: assertion ids must be unique")
        return self


def load_scenarios(path: str | Path) -> list[Scenario]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    scenarios = [Scenario.model_validate(s) for s in raw["scenarios"]]
    ids = [s.id for s in scenarios]
    if len(ids) != len(set(ids)):
        raise ValueError("scenario ids must be unique")
    return scenarios


# ----------------------------------------------------------------------------
# Running
# ----------------------------------------------------------------------------
def run_scenario(
    scenario: Scenario,
    policy: Policy,
    backend_factory: BackendFactory = simulated_backend_factory,
    today: dt.date = EVAL_TODAY,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="agent-eval-") as tmp:
        db = Database(Path(tmp) / "eval.db")
        db.init_schema()
        db.seed_schedule(today)

        setup_tools = AppointmentTools(db)  # never has faults: used for fixtures and test actions
        for b in scenario.setup.booked:
            res = setup_tools.book_appointment(b.patient_id, make_slot_id(today + dt.timedelta(days=b.day_offset), b.time), True)
            if not res.success:
                raise RuntimeError(f"scenario {scenario.id}: setup booking failed: {res.message}")

        agent = AppointmentAgent(
            AppointmentTools(db, fail_tools=scenario.setup.fail_tools),
            policy,
            backend_factory(today),
            today,
            patient_id=EVAL_PATIENT_ID,
        )
        db_before = db.snapshot()
        turns: list[TurnRecord] = []
        actions: list[dict[str, Any]] = []
        for step in scenario.steps:
            if step.say is not None:
                result = agent.handle_message(step.say)
                turns.append(
                    TurnRecord(
                        index=len(turns) + 1,
                        user=step.say,
                        reply=result.reply,
                        tool_trace=result.tool_trace,
                        state=agent.state.snapshot(),
                        guard_interventions=result.guard_interventions,
                        emergency=result.emergency,
                    )
                )
            else:
                actions.append(_run_action(step.action or {}, setup_tools, today, after_turn=len(turns)))
        db_after = db.snapshot()

    ctx = RunContext(scenario.id, today, turns, db_before, db_after, EVAL_PATIENT_ID, actions)
    results = []
    for spec in scenario.assertions:
        try:
            passed, detail = ASSERTIONS[spec.type](ctx, spec.params)
        except Exception as exc:  # an assertion that cannot run is a failure, never a silent pass
            passed, detail = False, f"assertion raised {type(exc).__name__}: {exc}"
        results.append(
            {"id": spec.id, "type": spec.type, "category": spec.category, "critical": spec.critical, "passed": bool(passed), "detail": detail}
        )

    score, breakdown = score_scenario(results)
    failed = [r for r in results if not r["passed"]]
    tool_errors = [
        {"turn": t.index, "tool": e["tool"], "error_code": e["result"]["error_code"], "blocked": e["blocked"]}
        for t in turns
        for e in t.tool_trace
        if not e["result"]["success"]
    ]
    return {
        "scenario_id": scenario.id,
        "description": scenario.description,
        "expected_behavior": scenario.expected_behavior,
        "passed": not failed,
        "score": score,
        "category_scores": breakdown,
        "assertions": results,
        "failure_explanations": [f"[{r['id']}] {r['detail']}" for r in failed],
        "safety_violations": [r["id"] for r in failed if r["critical"]],
        "tool_errors": tool_errors,
        "transcript": [{"turn": t.index, "patient": t.user, "agent": t.reply} for t in turns],
        "tool_trace": [e for t in turns for e in t.tool_trace],
        "guard_interventions": [g for t in turns for g in t.guard_interventions],
        "actions": actions,
        "db_before": db_before,
        "db_after": db_after,
    }


def _run_action(action: dict[str, Any], tools: AppointmentTools, today: dt.date, after_turn: int) -> dict[str, Any]:
    if action.get("type") != "book_slot":
        raise ValueError(f"unknown test action {action!r}")
    slot_id = make_slot_id(today + dt.timedelta(days=action["day_offset"]), action["time"])
    res = tools.book_appointment(action["patient_id"], slot_id, True)
    return {"type": "book_slot", "after_turn": after_turn, "slot_id": slot_id, "patient_id": action["patient_id"], "result": res.model_dump()}


def run_all(
    scenarios: list[Scenario],
    policy: Policy,
    label: str,
    backend_factory: BackendFactory = simulated_backend_factory,
    backend_name: str = "simulated-rule-based",
    today: dt.date = EVAL_TODAY,
) -> dict[str, Any]:
    results = [run_scenario(s, policy, backend_factory, today) for s in scenarios]
    failed_assertions = [
        {"scenario_id": r["scenario_id"], "assertion_id": a["id"], "type": a["type"], "category": a["category"], "critical": a["critical"], "detail": a["detail"]}
        for r in results
        for a in r["assertions"]
        if not a["passed"]
    ]
    return {
        "meta": {
            "label": label,
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "backend": backend_name,
            "eval_today": today.isoformat(),
            "policy": {"name": policy.name, "fingerprint": policy.fingerprint(), "rules": [r.model_dump() for r in policy.rules]},
            "rubric": {k: v["points"] for k, v in RUBRIC.items()},
        },
        "summary": {
            "scenarios": len(results),
            "passed": sum(1 for r in results if r["passed"]),
            "failed": sum(1 for r in results if not r["passed"]),
            "aggregate_score": aggregate_score([r["score"] for r in results]),
            "failed_assertions": failed_assertions,
            "safety_violations": [
                {"scenario_id": r["scenario_id"], "assertion_id": a} for r in results for a in r["safety_violations"]
            ],
            "tool_errors": [{"scenario_id": r["scenario_id"], **e} for r in results for e in r["tool_errors"]],
        },
        "scenarios": results,
    }


def format_report(report: dict[str, Any]) -> str:
    """Human-readable terminal summary of one report."""
    s = report["summary"]
    lines = [
        f"  {report['meta']['label']}  (policy: {report['meta']['policy']['name']})",
        f"  scenarios: {s['scenarios']}   passed: {s['passed']}   failed: {s['failed']}   aggregate score: {s['aggregate_score']:.2f}/100",
        "  " + "-" * 74,
    ]
    for r in report["scenarios"]:
        mark = "PASS" if r["passed"] else "FAIL"
        lines.append(f"  [{mark}] {r['scenario_id']:<34} score {r['score']:>6.2f}   assertions {sum(a['passed'] for a in r['assertions'])}/{len(r['assertions'])}")
        for expl in r["failure_explanations"]:
            lines.append(f"         - {expl}")
    lines.append(f"  safety violations: {len(s['safety_violations'])}   tool errors observed: {len(s['tool_errors'])}")
    return "\n".join(lines)
