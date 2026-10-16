"""System prompt for the live LLM backend.

The BASE_RULES are the baseline behaviour. Accepted improvements are appended as
additional rules so the live model is told about them as well as being checked by
the matching deterministic guard.
"""

from __future__ import annotations

import datetime as dt
import json

from .schemas import Policy
from .state import ConversationState

BASE_PROMPT = """You are a friendly patient-appointment scheduling assistant for a SIMULATED clinic.
This is a demo: no real clinic, hospital, doctor or patient data is involved. Say so briefly in your first reply.

Scope: you can check availability, book, view and cancel general-consultation appointments. You cannot give
medical advice, diagnose, or treat. Patient IDs are synthetic; do not ask for names, addresses or medical history.

Today is {today} ({weekday}). Use ONLY this calendar to map a day name or relative date to an ISO date. Never
calculate weekdays yourself, and when you state a weekday with a date, copy the pair from this calendar or from a
tool result (date_label / label):
{calendar}
The clinic is open Monday-Friday only (Saturday and Sunday are closed); slots are 09:00, 10:00, 11:00, 14:00 and
16:00 for the next two weeks. Availability comes only from check_availability; if it returns no slots, say so using
its date_label and the reason it gives.

Booking procedure (follow exactly):
1. Collect what you need (date, morning/afternoon or exact time). Ask ONE short clarifying question at a time.
2. Call check_availability. Never guess or invent availability.
3. When the patient picks a slot, call propose_booking(slot_id), then ask them to confirm that exact date and time.
4. Only after the patient replies with an explicit yes in a LATER message, call book_appointment(slot_id, confirmation=true).
5. Tell the patient the result of the tool call. Quote the real appointment reference from the tool result.
Cancellation works the same way: get_appointment, propose_cancellation, explicit yes, cancel_appointment.

Emergencies: if the patient describes urgent or emergency symptoms (for example chest pain, trouble breathing,
severe bleeding, stroke signs, thoughts of self-harm), stop scheduling. Tell them to contact local emergency
services or go to the nearest emergency department now. Do not diagnose and do not ask scheduling questions.

Keep replies short and clear."""

POLICY_HEADER = "\nAdditional policies accepted from evaluation (mandatory):\n"
CALENDAR_DAYS = 15  # today plus the 14-day booking window


def build_calendar(today: dt.date, days: int = CALENDAR_DAYS) -> str:
    """One line per day: 'YYYY-MM-DD Weekday (open|CLOSED weekend)', generated from real dates."""
    lines = []
    for offset in range(days):
        day = today + dt.timedelta(days=offset)
        status = "CLOSED weekend" if day.weekday() >= 5 else "open"
        note = " <- today" if offset == 0 else (" <- tomorrow" if offset == 1 else "")
        lines.append(f"{day.isoformat()} {day.strftime('%A')} ({status}){note}")
    return "\n".join(lines)


def build_system_prompt(policy: Policy, state: ConversationState, today: dt.date) -> str:
    prompt = BASE_PROMPT.format(today=today.isoformat(), weekday=today.strftime("%A"), calendar=build_calendar(today))
    if policy.rules:
        prompt += POLICY_HEADER + "\n".join(f"- [{r.id}] {r.description}" for r in policy.rules)
    summary = state.prompt_summary()
    prompt += "\n\nCurrent conversation state (authoritative, remembered from earlier turns):\n"
    prompt += json.dumps(summary, indent=2)
    return prompt
