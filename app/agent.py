"""The appointment agent: core safety invariants + policy guardrails around a backend.

Core invariants (always on, implemented here, not removable by improvements):

1. ``emergency_escalation``        - urgent symptoms are handled before any model/tool call.
2. ``explicit_confirmation_gate``  - book/cancel tools only run after a proposal in an
                                     earlier turn AND an explicit yes in the current message.
3. ``db_double_booking_constraint``- enforced by SQLite (see database.py).
4. ``simulation_disclosure``       - the first reply always tells the patient this is a simulation.

Policy guardrails (from ``Policy.rules``) are applied to the backend's draft reply.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from . import messages
from .errors import ProviderError
from .guardrails import apply_guardrails
from .nlu import detect_emergency, is_affirmative, says_not_emergency
from .prompts import build_system_prompt
from .schemas import Policy, ToolResult
from .state import ConversationState, PendingAction
from .tools import AppointmentTools
from .utils import correct_weekdays

class Backend(Protocol):
    """Anything that can turn a patient message into a draft reply (using tools)."""

    def generate_reply(
        self, user_message: str, state: ConversationState, executor: "ToolExecutor", system_prompt: str
    ) -> str: ...


@dataclass
class TurnResult:
    reply: str
    tool_trace: list[dict[str, Any]] = field(default_factory=list)
    guard_interventions: list[dict[str, Any]] = field(default_factory=list)
    emergency: bool = False
    provider_error: Optional[str] = None


class ToolExecutor:
    """Model-facing tool layer. Injects the patient ID, enforces the confirmation
    gate and records every call (including blocked ones) in the turn trace."""

    def __init__(self, tools: AppointmentTools, state: ConversationState, user_message: str):
        self.tools = tools
        self.state = state
        self.user_message = user_message
        self.trace: list[dict[str, Any]] = []

    # -- public entry point ---------------------------------------------
    def call(self, name: str, args: Optional[dict[str, Any]] = None) -> ToolResult:
        args = dict(args or {})
        handler = {
            "check_availability": self._check_availability,
            "propose_booking": self._propose_booking,
            "book_appointment": self._book,
            "get_appointment": self._get,
            "propose_cancellation": self._propose_cancellation,
            "cancel_appointment": self._cancel,
        }.get(name)
        if handler is None:
            result, blocked = ToolResult(success=False, error_code="UNKNOWN_TOOL", message=f"Unknown tool '{name}'."), True
        else:
            try:
                result, blocked = handler(args)
            except TypeError as exc:
                result, blocked = ToolResult(success=False, error_code="INVALID_INPUT", message=f"Invalid arguments: {exc}"), True
        self.trace.append(
            {"turn": self.state.turn, "tool": name, "args": args, "blocked": blocked, "result": result.model_dump()}
        )
        return result

    # -- handlers: return (result, blocked_by_gate) ------------------------
    def _check_availability(self, a: dict[str, Any]) -> tuple[ToolResult, bool]:
        res = self.tools.check_availability(a.get("date", ""), a.get("time_preference", "any"))
        if res.success:
            self.state.date = res.data["date"]
            self.state.time_preference = res.data["time_preference"] if res.data["time_preference"] != "any" else self.state.time_preference
            self.state.offered_slots = res.data["slots"]
        return res, False

    def _propose_booking(self, a: dict[str, Any]) -> tuple[ToolResult, bool]:
        res = self.tools.describe_slot(a.get("slot_id", ""))
        if res.success:
            self.state.pending_action = PendingAction("book", res.data["slot_id"], self.state.turn, res.data["label"])
            res = ToolResult(
                success=True,
                message=(
                    f"Slot open: {res.data['label']}. Ask the patient to explicitly confirm this exact slot. "
                    "Do not book until they reply yes in their next message."
                ),
                data=res.data,
            )
        return res, False

    def _gate_ok(self, kind: str, target: str, confirmation: Any) -> bool:
        pa = self.state.pending_action
        return bool(
            pa is not None
            and pa.type == kind
            and pa.target_id == target
            and self.state.turn > pa.proposed_turn
            and confirmation is True
            and is_affirmative(self.user_message)
        )

    def _book(self, a: dict[str, Any]) -> tuple[ToolResult, bool]:
        slot_id = a.get("slot_id", "")
        if not self._gate_ok("book", slot_id, a.get("confirmation")):
            return (
                ToolResult(
                    success=False,
                    error_code="CONFIRMATION_REQUIRED",
                    message="Blocked: the patient has not explicitly confirmed this exact slot in their latest message.",
                ),
                True,
            )
        res = self.tools.book_appointment(self.state.patient_id, slot_id, True)
        self.state.pending_action = None
        if res.success:
            self.state.last_booked_appointment_id = res.data["appointment_id"]
        return res, False

    def _get(self, a: dict[str, Any]) -> tuple[ToolResult, bool]:
        return self.tools.get_appointment(a.get("appointment_id", "")), False

    def _propose_cancellation(self, a: dict[str, Any]) -> tuple[ToolResult, bool]:
        res = self.tools.get_appointment(a.get("appointment_id", ""))
        if res.success and res.data["status"] != "booked":
            return ToolResult(success=False, error_code="ALREADY_CANCELLED", message="That appointment is already cancelled."), False
        if res.success:
            self.state.pending_action = PendingAction(
                "cancel", res.data["appointment_id"], self.state.turn, f"{res.data['appointment_id']} ({res.data['label']})"
            )
            res = ToolResult(
                success=True,
                message="Appointment found. Ask the patient to explicitly confirm cancelling it. Do not cancel until they reply yes next message.",
                data=res.data,
            )
        return res, False

    def _cancel(self, a: dict[str, Any]) -> tuple[ToolResult, bool]:
        appt_id = a.get("appointment_id", "")
        if not self._gate_ok("cancel", appt_id, a.get("confirmation")):
            return (
                ToolResult(
                    success=False,
                    error_code="CONFIRMATION_REQUIRED",
                    message="Blocked: the patient has not explicitly confirmed this cancellation in their latest message.",
                ),
                True,
            )
        res = self.tools.cancel_appointment(appt_id, True)
        self.state.pending_action = None
        return res, False


class AppointmentAgent:
    def __init__(
        self,
        tools: AppointmentTools,
        policy: Policy,
        backend: Backend,
        today: dt.date,
        patient_id: str = "PAT-1001",
    ):
        if not policy.core_invariants_intact():
            raise ValueError("Policy is missing required core invariants; refusing to start.")
        self.tools = tools
        self.policy = policy
        self.backend = backend
        self.today = today
        self.state = ConversationState(patient_id=patient_id)

    # ------------------------------------------------------------------
    def handle_message(self, text: str) -> TurnResult:
        state = self.state
        state.turn += 1
        text = (text or "").strip()
        state.history.append({"role": "user", "text": text})

        result = self._emergency_turn(text)
        if result is None:
            result = self._normal_turn(text)

        state.history.append({"role": "assistant", "text": result.reply})
        return result

    # -- core invariant 1: emergencies ------------------------------------
    def _emergency_turn(self, text: str) -> Optional[TurnResult]:
        state = self.state
        if detect_emergency(text):
            state.emergency_flagged = True
            state.pending_action = None
            return TurnResult(self._with_disclosure(messages.EMERGENCY_MESSAGE), emergency=True)
        if state.emergency_flagged:
            if says_not_emergency(text):
                state.emergency_flagged = False
                return TurnResult(messages.EMERGENCY_CLEARED, emergency=False)
            return TurnResult(messages.EMERGENCY_REMINDER, emergency=True)
        return None

    # -- normal scheduling turn ------------------------------------------
    def _normal_turn(self, text: str) -> TurnResult:
        state = self.state
        executor = ToolExecutor(self.tools, state, text)
        provider_error: Optional[str] = None
        try:
            draft = self.backend.generate_reply(
                text, state, executor, build_system_prompt(self.policy, state, self.today)
            )
        except ProviderError as exc:
            provider_error = f"{exc.kind}: {exc.user_message}"
            draft = self._provider_fallback(executor.trace, exc)

        reply, interventions = apply_guardrails(self.policy.rules, draft, executor.trace, state, self.tools)
        reply, weekday_fixes = correct_weekdays(reply, self.today)  # always on, independent of the policy
        for fix in weekday_fixes:
            interventions.append({"rule_id": "CORE-WEEKDAY-CONSISTENCY", "enforcement": "correct_weekdays", "reason": fix, "original_reply": draft})
        reply = self._with_disclosure(reply)
        return TurnResult(reply, executor.trace, interventions, False, provider_error)

    def _provider_fallback(self, trace: list[dict[str, Any]], exc: ProviderError) -> str:
        """Report only what the tools actually did when the model call failed."""
        for entry in reversed(trace):
            res = entry["result"]
            if entry["blocked"] or not res["success"]:
                continue
            if entry["tool"] == "book_appointment":
                return messages.booking_confirmed(res["data"]["label"], res["data"]["appointment_id"])
            if entry["tool"] == "cancel_appointment":
                return messages.cancellation_confirmed(res["data"]["appointment_id"], res["data"]["label"])
        return messages.provider_unavailable(exc.user_message)

    # -- core invariant 4: simulation disclosure ----------------------------
    def _with_disclosure(self, reply: str) -> str:
        if self.state.turn == 1 and "simulated" not in reply.lower():
            return f"{messages.DISCLOSURE} {reply}"
        return reply
