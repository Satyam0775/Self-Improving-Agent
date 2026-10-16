"""Policy guardrails: deterministic checks applied to a draft reply before it is shown.

A policy rule never contains code. It only *names* a guard from ``GUARD_REGISTRY``
(plus constrained string params). Guards are implemented here, reviewed by humans,
and may only inspect the draft reply, this turn's tool trace and the database.
An improvement can therefore enable behaviour that already exists in code, but it
cannot add arbitrary logic or remove the core invariants (those live in agent.py).
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional

from . import messages
from .schemas import PolicyRule
from .state import ConversationState
from .tools import AppointmentTools


@dataclass
class GuardContext:
    reply: str
    trace: list[dict[str, Any]]  # tool trace entries of THIS turn
    state: ConversationState
    tools: AppointmentTools
    rule: PolicyRule


@dataclass
class GuardOutcome:
    reply: str
    reason: str


GuardFn = Callable[[GuardContext], Optional[GuardOutcome]]


@dataclass
class RegisteredGuard:
    fn: GuardFn
    allowed_params: dict[str, set[str]]  # param name -> allowed values


GUARD_REGISTRY: dict[str, RegisteredGuard] = {}


def register_guard(name: str, allowed_params: Optional[dict[str, set[str]]] = None):
    def decorator(fn: GuardFn) -> GuardFn:
        GUARD_REGISTRY[name] = RegisteredGuard(fn, allowed_params or {})
        return fn

    return decorator


@contextmanager
def temporary_guard(name: str, fn: GuardFn, allowed_params: Optional[dict[str, set[str]]] = None) -> Iterator[None]:
    """Register a guard for the duration of a ``with`` block (used by tests / negative control)."""
    previous = GUARD_REGISTRY.get(name)
    GUARD_REGISTRY[name] = RegisteredGuard(fn, allowed_params or {})
    try:
        yield
    finally:
        if previous is None:
            GUARD_REGISTRY.pop(name, None)
        else:
            GUARD_REGISTRY[name] = previous


# -- claim detection -----------------------------------------------------------
_SENTENCES = re.compile(r"(?<=[.!?])\s+|\n+")
_NEGATION = re.compile(
    r"\b(not|no|never|unable|couldn'?t|could not|can'?t|cannot|failed|fail|without|unfortunately|"
    r"already|status|if|would you like|shall i|should i|can i|may i|once)\b|n't"
)
_BOOKING_CLAIM = re.compile(r"\b(confirmed|booked|scheduled|all set|reserved|secured)\b|see you (then|on|at)|you'?re (set|good to go)")
_CANCEL_CLAIM = re.compile(r"\b(cancelled|canceled)\b")


def _claims(text: str, pattern: re.Pattern[str]) -> bool:
    for sentence in _SENTENCES.split(text):
        s = sentence.lower().strip()
        if not s or s.endswith("?") or _NEGATION.search(s):
            continue
        if pattern.search(s):
            return True
    return False


def claims_booking_success(text: str) -> bool:
    return _claims(text, _BOOKING_CLAIM)


def claims_cancellation_success(text: str) -> bool:
    return _claims(text, _CANCEL_CLAIM)


# -- built-in guards ---------------------------------------------------------------
_ACTIONS = {
    "booking": {"tool": "book_appointment", "claim": claims_booking_success, "status": "booked"},
    "cancellation": {"tool": "cancel_appointment", "claim": claims_cancellation_success, "status": "cancelled"},
}


@register_guard("verify_action_claims", allowed_params={"action": {"booking", "cancellation"}})
def verify_action_claims(ctx: GuardContext) -> Optional[GuardOutcome]:
    """Block unsupported success claims.

    A reply may claim a booking/cancellation only if (1) the corresponding tool ran
    this turn and returned success=true AND (2) the database confirms the resulting
    appointment status. Otherwise the reply is replaced with an honest failure message.
    """
    action = ctx.rule.params.get("action", "booking")
    cfg = _ACTIONS[action]
    if not cfg["claim"](ctx.reply):
        return None

    executed = [e for e in ctx.trace if e["tool"] == cfg["tool"] and not e.get("blocked")]
    failed_msg = ""
    if executed and not executed[-1]["result"]["success"]:
        failed_msg = executed[-1]["result"]["message"]
    successes = [e for e in executed if e["result"]["success"]]
    fail = messages.booking_failed if action == "booking" else messages.cancellation_failed

    if not successes:
        detail = failed_msg or "I did not receive a successful confirmation from the scheduling system."
        return GuardOutcome(fail(detail), f"claimed {action} success without a successful {cfg['tool']} result")

    appt_id = successes[-1]["result"]["data"].get("appointment_id")
    truth = ctx.tools.verify_appointment(appt_id) if appt_id else None
    if truth is None or truth["status"] != cfg["status"]:
        return GuardOutcome(
            fail("The database does not show that change."),
            f"{action} claimed but database state for {appt_id} does not match",
        )
    if action == "booking":
        cited = set(re.findall(r"APT-\d{5}", ctx.reply))
        if cited and cited != {appt_id}:
            return GuardOutcome(
                messages.booking_confirmed(truth_label(truth), appt_id),
                f"reply cited reference(s) {sorted(cited)} but the verified booking is {appt_id}",
            )
    return None


def truth_label(appt: dict[str, Any]) -> str:
    from .utils import slot_label

    return slot_label(appt["date"], appt["time"])


# -- application ------------------------------------------------------------------
def apply_guardrails(
    rules: list[PolicyRule],
    reply: str,
    trace: list[dict[str, Any]],
    state: ConversationState,
    tools: AppointmentTools,
) -> tuple[str, list[dict[str, Any]]]:
    interventions: list[dict[str, Any]] = []
    for rule in rules:
        guard = GUARD_REGISTRY.get(rule.enforcement)
        if guard is None:
            raise KeyError(f"Policy rule {rule.id} names unknown guard '{rule.enforcement}'")
        outcome = guard.fn(GuardContext(reply, trace, state, tools, rule))
        if outcome is not None:
            interventions.append(
                {"rule_id": rule.id, "enforcement": rule.enforcement, "reason": outcome.reason, "original_reply": reply}
            )
            reply = outcome.reply
    return reply, interventions
