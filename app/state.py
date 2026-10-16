"""Conversation state that persists across turns of one patient session."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class PendingAction:
    """A booking/cancellation the patient has been asked to confirm."""

    type: str  # "book" | "cancel"
    target_id: str  # slot_id or appointment_id
    proposed_turn: int
    description: str


@dataclass
class ConversationState:
    patient_id: str
    turn: int = 0
    intent: Optional[str] = None  # "book" | "cancel" | "view"
    consultation_type: Optional[str] = None
    asked_consultation_type: bool = False
    date: Optional[str] = None  # ISO date chosen so far
    time_preference: Optional[str] = None
    requested_time: Optional[str] = None  # 'HH:MM'
    offered_slots: list[dict[str, Any]] = field(default_factory=list)
    pending_action: Optional[PendingAction] = None
    appointment_id: Optional[str] = None
    last_booked_appointment_id: Optional[str] = None
    emergency_flagged: bool = False
    history: list[dict[str, str]] = field(default_factory=list)

    def reset_task(self) -> None:
        """Clear task-specific fields after a task finishes (history is kept)."""
        self.intent = None
        self.consultation_type = None
        self.asked_consultation_type = False
        self.date = None
        self.time_preference = None
        self.requested_time = None
        self.offered_slots = []
        self.pending_action = None
        self.appointment_id = None

    def snapshot(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("history")
        return data

    def prompt_summary(self) -> dict[str, Any]:
        """Compact view injected into the live model's system prompt."""
        snap = self.snapshot()
        snap["offered_slots"] = [s.get("slot_id") for s in self.offered_slots]
        return {k: v for k, v in snap.items() if v not in (None, [], False)}
