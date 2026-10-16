"""Deterministic, rule-based stand-in for an LLM.

It lets the evaluation run offline and reproducibly. It follows the same contract
as the live backend (``generate_reply`` + tools through the ``ToolExecutor``).

IMPORTANT, documented limitation: with ``optimistic_booking_claims=True`` it
deliberately reproduces a known LLM failure mode - after calling the booking tool it
writes a success message *without checking the tool result*. That is the baseline
failure the improvement loop detects and fixes with a policy guardrail. It is a
test double for a failure mode, not a claim about how any specific live model behaves.
"""

from __future__ import annotations

import datetime as dt
from typing import Optional

from . import messages
from .agent import ToolExecutor
from .nlu import Entities, detect_intent, extract_entities, is_affirmative, is_negative
from .state import ConversationState
from .utils import format_date, format_time, join_natural, slot_label

HELP_TEXT = (
    "Hello! I can help you book, view or cancel an appointment. This is a simulated scheduling demo, "
    "not a real clinic. What would you like to do?"
)


class SimulatedModel:
    name = "simulated-rule-based"

    def __init__(self, today: dt.date, optimistic_booking_claims: bool = True):
        self.today = today
        self.optimistic_booking_claims = optimistic_booking_claims

    # ------------------------------------------------------------------
    def generate_reply(
        self, user_message: str, state: ConversationState, executor: ToolExecutor, system_prompt: str = ""
    ) -> str:
        ents = extract_entities(user_message, self.today)
        has_new_info = bool(ents.date or ents.time or ents.time_preference)

        pending = state.pending_action
        if pending is not None and state.turn > pending.proposed_turn:
            if is_affirmative(user_message):
                return self._execute_pending(state, executor)
            if is_negative(user_message) and not has_new_info:
                state.pending_action = None
                if pending.type == "book":
                    state.requested_time = None
                    return (
                        "No problem. I have not booked anything. Would you like to choose a different "
                        "time, or is there something else I can help with?"
                    )
                return f"Understood. I have not cancelled anything; appointment {pending.target_id} is unchanged."
            state.pending_action = None  # the patient is changing details; carry on

        intent = detect_intent(user_message)
        if intent:
            if intent == "book" and state.intent not in (None, "book"):
                state.reset_task()
            state.intent = intent
        elif state.intent is None and (ents.date or ents.time or ents.time_preference):
            state.intent = "book"

        if state.intent == "cancel":
            return self._cancel_flow(state, ents, executor)
        if state.intent == "view":
            return self._view_flow(state, ents, executor)
        if state.intent == "book":
            return self._booking_flow(state, ents, executor)
        return HELP_TEXT

    # -- booking ----------------------------------------------------------
    def _booking_flow(self, state: ConversationState, ents: Entities, ex: ToolExecutor) -> str:
        if ents.date and ents.date != state.date:
            state.date = ents.date
            state.offered_slots = []
        if ents.time_preference:
            state.time_preference = ents.time_preference
        if ents.time:
            state.requested_time = ents.time

        if state.date is None:
            if ents.ambiguous_date:
                return (
                    "I'd be happy to help. To look for openings I need a specific day, for example "
                    "'tomorrow' or 'Thursday'. Which day works for you?"
                )
            return "Certainly. Which day would you like the appointment? You can say a day like 'tomorrow' or 'Thursday'."

        note = ""
        if state.consultation_type is None:
            if ents.type_mentioned:
                state.consultation_type = "general consultation"
            elif state.asked_consultation_type:
                state.consultation_type = "general consultation"
                note = "This demo only offers general consultations, so I'll use that. "
            else:
                state.asked_consultation_type = True
                return "Certainly. What type of consultation are you looking for? This demo offers general consultations."

        date_label = format_date(dt.date.fromisoformat(state.date))
        if state.time_preference is None and state.requested_time is None:
            return f"{note}Would morning or afternoon work better on {date_label}?"

        pref = "any" if state.requested_time else (state.time_preference or "any")
        avail = ex.call("check_availability", {"date": state.date, "time_preference": pref})
        if not avail.success:
            return f"{note}I couldn't check availability right now ({avail.message}). Please try again or choose another day."
        slots = avail.data["slots"]

        if state.requested_time:
            wanted = state.requested_time
            match = [s for s in slots if s["time"] == wanted]
            if match:
                prop = ex.call("propose_booking", {"slot_id": match[0]["slot_id"]})
                if prop.success:
                    state.requested_time = None
                    return (
                        f"{note}I can book a {state.consultation_type} on {date_label} at {format_time(wanted)}. "
                        "Would you like me to confirm this appointment? (yes/no)"
                    )
            state.requested_time = None
            if slots:
                times = join_natural([format_time(s["time"]) for s in slots])
                return (
                    f"{note}I'm sorry, {format_time(wanted)} on {date_label} is not available. "
                    f"The open times that day are {times}. Which would you like?"
                )
            state.date = None
            return (
                f"{note}I'm sorry, {format_time(wanted)} on {date_label} is not available and there are no other "
                "openings that day. Would you like to try a different day?"
            )

        if slots:
            times = join_natural([format_time(s["time"]) for s in slots])
            return f"{note}I have {times} available on {date_label}. Which time would you like?"
        weekend = dt.date.fromisoformat(state.date).weekday() >= 5
        state.date = None
        state.time_preference = None
        hint = " The clinic is closed on weekends." if weekend else ""
        return f"{note}I'm sorry, there are no openings for that on {date_label}.{hint} Would you like to try a different day?"

    # -- executing a confirmed action ------------------------------------------
    def _execute_pending(self, state: ConversationState, ex: ToolExecutor) -> str:
        pending = state.pending_action
        assert pending is not None
        if pending.type == "book":
            res = ex.call("book_appointment", {"slot_id": pending.target_id, "confirmation": True})
            if res.success:
                reply = messages.booking_confirmed(res.data["label"], res.data["appointment_id"])
                state.reset_task()
                return reply
            if self.optimistic_booking_claims:
                # Known LLM failure mode reproduced on purpose: claims success without checking the result.
                label = pending.description
                state.reset_task()
                return f"Your appointment is confirmed for {label}. You're all set! This is a simulated booking."
            state.pending_action = None
            return messages.booking_failed(res.message)
        res = ex.call("cancel_appointment", {"appointment_id": pending.target_id, "confirmation": True})
        if res.success:
            reply = messages.cancellation_confirmed(res.data["appointment_id"], res.data["label"])
            state.reset_task()
            return reply
        state.pending_action = None
        return messages.cancellation_failed(res.message)

    # -- cancel / view ------------------------------------------------------------
    def _cancel_flow(self, state: ConversationState, ents: Entities, ex: ToolExecutor) -> str:
        ref = ents.appointment_id or state.appointment_id
        if not ref:
            return "I can help with that. Please share the reference of the appointment to cancel (it looks like APT-00001)."
        state.appointment_id = ref
        found = ex.call("get_appointment", {"appointment_id": ref})
        if not found.success:
            state.appointment_id = None
            return (
                f"I couldn't find that appointment ({found.message}). I haven't cancelled anything. "
                "Please double-check the reference; it looks like APT-00001."
            )
        if found.data["status"] != "booked":
            state.reset_task()
            return f"Appointment {ref} is already cancelled, so there is nothing more to do."
        prop = ex.call("propose_cancellation", {"appointment_id": ref})
        if not prop.success:
            return f"I couldn't prepare that cancellation ({prop.message}). Nothing has been changed."
        return (
            f"I found appointment {ref}: {found.data['label']}. "
            "Would you like me to cancel it? (yes/no)"
        )

    def _view_flow(self, state: ConversationState, ents: Entities, ex: ToolExecutor) -> str:
        ref = ents.appointment_id or state.appointment_id
        if not ref:
            return "Sure. Which appointment? Please share the reference (it looks like APT-00001)."
        found = ex.call("get_appointment", {"appointment_id": ref})
        state.reset_task()
        if not found.success:
            return f"I couldn't find that appointment ({found.message}). Please double-check the reference."
        d = found.data
        return f"Appointment {ref}: {d['consultation_type']} on {slot_label(d['date'], d['time'])}. Current status: {d['status']}."
