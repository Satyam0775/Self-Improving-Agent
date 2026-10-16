"""Pydantic models: tool inputs/outputs, policies and improvement records."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator

PATIENT_ID_RE = r"^PAT-\d{4,6}$"
SLOT_ID_RE = r"^SLOT-\d{8}-\d{4}$"
APPOINTMENT_ID_RE = r"^APT-\d{5}$"

# Safety invariants that are implemented in code and can never be removed by an
# improvement. The evaluator re-checks that they are declared on every policy.
CORE_INVARIANTS = [
    "emergency_escalation",  # urgent symptoms -> urgent-care advice, no routine booking
    "explicit_confirmation_gate",  # no book/cancel without a prior proposal + explicit yes
    "db_double_booking_constraint",  # SQLite unique index + transactions
    "simulation_disclosure",  # patient is told this is a simulated clinic
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# ----------------------------------------------------------------------------
# Tools
# ----------------------------------------------------------------------------
class ToolResult(StrictModel):
    """Structured result returned by every tool."""

    success: bool
    error_code: Optional[str] = None
    message: str
    data: dict[str, Any] = Field(default_factory=dict)


class CheckAvailabilityInput(StrictModel):
    date: str
    time_preference: Literal["morning", "afternoon", "any"] = "any"

    @field_validator("date")
    @classmethod
    def _iso_date(cls, v: str) -> str:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
            raise ValueError("must be an ISO date in the form YYYY-MM-DD")
        try:
            dt.date.fromisoformat(v)
        except ValueError:
            raise ValueError("is not a real calendar date") from None
        return v


class BookInput(StrictModel):
    patient_id: str = Field(pattern=PATIENT_ID_RE)
    slot_id: str = Field(pattern=SLOT_ID_RE)
    confirmation: StrictBool


class GetAppointmentInput(StrictModel):
    appointment_id: str = Field(pattern=APPOINTMENT_ID_RE)


class CancelInput(StrictModel):
    appointment_id: str = Field(pattern=APPOINTMENT_ID_RE)
    confirmation: StrictBool


# ----------------------------------------------------------------------------
# Policy
# ----------------------------------------------------------------------------
class PolicyRule(BaseModel):
    """One enforceable rule. ``enforcement`` names a guard implemented in code."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    id: str = Field(pattern=r"^GR-[A-Z0-9-]{3,40}$")
    description: str = Field(min_length=10, max_length=500)
    enforcement: str = Field(min_length=3, max_length=60)
    params: dict[str, str] = Field(default_factory=dict)


class Policy(StrictModel):
    name: str
    rules: list[PolicyRule] = Field(default_factory=list)
    core_invariants: list[str] = Field(default_factory=lambda: list(CORE_INVARIANTS))

    def with_rule(self, rule: PolicyRule) -> "Policy":
        """Return a NEW policy including ``rule``; ``self`` is never mutated."""
        return self.model_copy(
            deep=True,
            update={"rules": [*self.rules, rule], "name": f"{self.name}+{rule.id}"},
        )

    def fingerprint(self) -> str:
        payload = json.dumps(self.model_dump(), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:12]

    def core_invariants_intact(self) -> bool:
        return set(CORE_INVARIANTS).issubset(self.core_invariants)


# ----------------------------------------------------------------------------
# Improvement records
# ----------------------------------------------------------------------------
class Target(StrictModel):
    scenario_id: str
    assertion_id: str


class ImprovementRecord(StrictModel):
    """Auditable record of one proposed (and later accepted/rejected) improvement."""

    id: str = Field(pattern=r"^IMP-\d{3}$")
    type: Literal["guardrail"]
    source_scenario: str
    targets: list[Target] = Field(min_length=1)
    failure_description: str = Field(min_length=10)
    rule: str = Field(min_length=10, max_length=500)
    rule_id: str = Field(pattern=r"^GR-[A-Z0-9-]{3,40}$")
    enforcement: str = Field(min_length=3, max_length=60)
    params: dict[str, str] = Field(default_factory=dict)
    status: Literal["proposed", "accepted", "rejected"] = "proposed"
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    baseline_score: Optional[float] = None
    candidate_score: Optional[float] = None
    regressions: list[str] = Field(default_factory=list)
    decision_reason: str = ""

    def to_policy_rule(self) -> PolicyRule:
        return PolicyRule(
            id=self.rule_id,
            description=self.rule,
            enforcement=self.enforcement,
            params=dict(self.params),
        )
