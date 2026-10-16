"""Groq backend tested against a scripted fake client (no network, no API key)."""

import json
from types import SimpleNamespace

import pytest

from app.agent import AppointmentAgent
from app.errors import ProviderError
from app.groq_backend import GroqBackend, build_tool_declarations
from app.policy import BASELINE_POLICY
from app.tools import MODEL_TOOL_SPECS, AppointmentTools

SECRET = "FAKE-SECRET-KEY-123"


def call(name, **args):
    return SimpleNamespace(id=f"call-{name}", function=SimpleNamespace(name=name, arguments=json.dumps(args)))


def calling(*calls):
    message = SimpleNamespace(content=None, tool_calls=list(calls))
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def saying(text):
    message = SimpleNamespace(content=text, tool_calls=[])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _FakeCompletions:
    def __init__(self, owner):
        self._owner = owner

    def create(self, **kwargs):
        self._owner.requests.append(kwargs)
        item = self._owner.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeClient:
    """Plays back scripted responses (or raises an exception) and records each request."""

    def __init__(self, *script):
        self.script, self.requests = list(script), []
        self.chat = SimpleNamespace(completions=_FakeCompletions(self))


def make(db, today, client, policy=BASELINE_POLICY):
    backend = GroqBackend(client, "test-model", api_key=SECRET)
    return AppointmentAgent(AppointmentTools(db), policy, backend, today, patient_id="PAT-1001"), backend


def test_tool_declarations_match_specs_and_do_not_expose_patient_id():
    decls = build_tool_declarations()
    assert [d["function"]["name"] for d in decls] == [s["name"] for s in MODEL_TOOL_SPECS]
    for d in decls:
        assert "patient_id" not in d["function"]["parameters"]["properties"]


def test_full_booking_conversation_through_function_calling(db, today, slot):
    client = FakeClient(
        calling(call("check_availability", date="2026-03-03", time_preference="afternoon")),
        calling(call("propose_booking", slot_id=slot(1, "16:00"))),
        saying("I can book 4:00 PM tomorrow. Shall I confirm it?"),
        calling(call("book_appointment", slot_id=slot(1, "16:00"), confirmation=True)),
        saying("Your appointment is confirmed. Your reference is APT-00001."),
    )
    agent, _ = make(db, today, client)
    first = agent.handle_message("I need an afternoon appointment tomorrow")
    assert "confirm" in first.reply and db.snapshot()["appointments"] == []
    second = agent.handle_message("Yes please")
    assert "APT-00001" in second.reply
    assert db.snapshot()["appointments"][0]["slot_id"] == slot(1, "16:00")

    fed_back = [m for r in client.requests for m in r["messages"] if m.get("role") == "tool"]
    assert fed_back and json.loads(fed_back[0]["content"])["result"]["success"] is True
    assert any(m.get("role") == "assistant" for m in client.requests[3]["messages"])
    assert "2026-03-03" in client.requests[3]["messages"][0]["content"]


def test_model_cannot_book_without_proposal_or_confirmation(db, today, slot):
    client = FakeClient(
        calling(call("book_appointment", slot_id=slot(1, "16:00"), confirmation=True)),
        saying("Sorry, I need your confirmation first. Shall I book 4 PM tomorrow?"),
    )
    agent, _ = make(db, today, client)
    res = agent.handle_message("book 4pm tomorrow")
    assert db.snapshot()["appointments"] == []
    assert res.tool_trace[0]["blocked"] and res.tool_trace[0]["result"]["error_code"] == "CONFIRMATION_REQUIRED"


def test_request_contains_tools_and_system_prompt(db, today):
    client = FakeClient(saying("Hello!"))
    agent, _ = make(db, today, client)
    agent.handle_message("hi")
    req = client.requests[0]
    assert req["tool_choice"] == "auto"
    assert req["tools"] and "SIMULATED" in req["messages"][0]["content"] and "Booking procedure" in req["messages"][0]["content"]


class ApiError(Exception):
    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code


@pytest.mark.parametrize(
    "exc,kind",
    [
        (ApiError(429, "quota"), "quota"),
        (ApiError(403, f"bad key {SECRET}"), "auth"),
        (ApiError(404, "model"), "model_not_found"),
        (ApiError(503, "down"), "server_error"),
        (TimeoutError(f"timed out calling with {SECRET}"), "timeout"),
        (ConnectionError(f"dns failure for {SECRET}"), "network"),
    ],
)
def test_provider_errors_are_handled_gracefully_without_leaking_the_key(db, today, exc, kind):
    agent, _ = make(db, today, FakeClient(exc))
    res = agent.handle_message("book me tomorrow at 4 pm")
    assert res.provider_error and res.provider_error.startswith(kind)
    assert SECRET not in res.reply and SECRET not in res.provider_error
    assert "confirmed" not in res.reply.lower() and db.snapshot()["appointments"] == []


def test_empty_model_reply_is_a_provider_error_not_a_blank_message(db, today):
    agent, _ = make(db, today, FakeClient(saying("   ")))
    res = agent.handle_message("hello")
    assert res.provider_error and res.reply.strip()


def test_runaway_tool_loop_is_stopped(db, today):
    agent, _ = make(db, today, FakeClient(*[calling(call("check_availability", date="2026-03-03")) for _ in range(10)]))
    res = agent.handle_message("hello")
    assert res.provider_error and res.provider_error.startswith("tool_loop")


def test_emergency_never_reaches_groq(db, today):
    client = FakeClient()  # empty script: any call would raise IndexError
    agent, _ = make(db, today, client)
    assert agent.handle_message("I have severe chest pain").emergency and client.requests == []


def test_guardrail_also_protects_live_model_output(db, today, slot):
    from app.schemas import PolicyRule

    rule = PolicyRule(id="GR-VERIFY-BOOKING-CLAIM", description="Verify booking claims before sending.", enforcement="verify_action_claims", params={"action": "booking"})
    client = FakeClient(
        calling(call("propose_booking", slot_id=slot(1, "16:00"))),
        saying("Confirm 4 PM?"),
        calling(call("book_appointment", slot_id=slot(1, "16:00"), confirmation=True)),
        saying("Great news, your appointment is confirmed!"),
    )
    db_fail = AppointmentTools(db, fail_tools={"book_appointment": "SERVICE_UNAVAILABLE"})
    backend = GroqBackend(client, "m")
    agent = AppointmentAgent(db_fail, BASELINE_POLICY.with_rule(rule), backend, today)
    agent.handle_message("book 4pm tomorrow")
    res = agent.handle_message("yes")
    assert "confirmed" not in res.reply.lower() and res.guard_interventions and db.snapshot()["appointments"] == []


def test_malformed_tool_arguments_do_not_crash_and_are_rejected(db, today):
    malformed = SimpleNamespace(id="c1", function=SimpleNamespace(name="check_availability", arguments="{bad-json"))
    client = FakeClient(calling(malformed), saying("ok"))
    agent, _ = make(db, today, client)
    res = agent.handle_message("tomorrow afternoon")
    assert res.tool_trace and res.tool_trace[0]["result"]["error_code"] == "INVALID_INPUT"
