"""Baseline policy and loading/saving of improvement records.

* ``BASELINE_POLICY`` is defined in code and is never modified. It is deliberately
  minimal: it contains no response-verification guardrail.
* ``improvements/accepted.json`` holds improvements that passed the acceptance
  gate. The *active* policy used by the live agent is baseline + accepted rules.
* The evaluation always starts from ``BASELINE_POLICY`` (not the active policy),
  so repeated runs are reproducible and never contaminated by earlier acceptances.
"""

from __future__ import annotations

import json
from pathlib import Path

from .schemas import ImprovementRecord, Policy

BASELINE_POLICY = Policy(name="baseline", rules=[])

ACCEPTED_FILE = "accepted.json"
PROPOSALS_FILE = "proposals.json"


def load_records(path: str | Path) -> list[ImprovementRecord]:
    p = Path(path)
    if not p.exists():
        return []
    raw = json.loads(p.read_text(encoding="utf-8") or "{}")
    return [ImprovementRecord.model_validate(item) for item in raw.get("improvements", [])]


def save_records(path: str | Path, records: list[ImprovementRecord]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"improvements": [r.model_dump(mode="json") for r in records]}
    p.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")


def upsert_record(records: list[ImprovementRecord], record: ImprovementRecord) -> list[ImprovementRecord]:
    """Replace the record with the same id, or append (keeps files idempotent)."""
    out = [r for r in records if r.id != record.id]
    out.append(record)
    return sorted(out, key=lambda r: r.id)


def load_active_policy(improvements_dir: str | Path) -> Policy:
    """Baseline policy + every *accepted* improvement from ``accepted.json``."""
    policy = BASELINE_POLICY.model_copy(deep=True)
    for record in load_records(Path(improvements_dir) / ACCEPTED_FILE):
        if record.status == "accepted" and all(r.id != record.rule_id for r in policy.rules):
            policy = policy.with_rule(record.to_policy_rule())
    return policy
