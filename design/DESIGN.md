# Design Note - Self-Improving Appointment Agent

*A simulated clinic demonstration, not a production healthcare system. No real patient data is used.*

## Problem and architecture
A patient chats in plain language to book, view or cancel an appointment. `AppointmentAgent` (`app/agent.py`) keeps
multi-turn state (`state.py`), sends the conversation to a backend, executes the model's tool calls through a single
`ToolExecutor`, and applies policy guardrails to the draft reply. Tools (`tools.py`) read and write SQLite
(`database.py`): a partial unique index plus `BEGIN IMMEDIATE` transactions make double booking impossible at the
database level. The **live backend** is Groq (`groq_backend.py`, chat completions with tool calling, model
`openai/gpt-oss-120b`, set as an internal default in `config.py`). An **offline rule-based backend**
(`simulated_model.py`) drives the evaluation harness. The harness (`evals/`) runs 13 YAML scenarios, scores them against
ground truth, finds failures, drafts a structured improvement, re-runs the *same* scenarios and accepts or rejects it.

## Key design choices
- **The LLM never decides outcomes.** It understands language and picks tools; success is whatever the tool and database
  say. The patient ID is injected by code, so the model cannot book for someone else.
- **Separation of concerns.** Agent loop, tools, storage, guardrails, evaluation and improvement acceptance are separate
  modules, so each is testable alone and the evaluation reuses the real agent, tools and database.
- **Safety in code, improvements as data.** Core invariants (emergency handling, confirmation gate, DB constraint,
  simulation disclosure) are code and cannot be removed. An improvement can only *enable a named guard that already
  exists* with constrained parameters, so a bad proposal cannot edit files or weaken safety.
- **Deterministic evaluation:** fixed date, fresh temp DB per scenario, no LLM judge, no network.

## Safety and limitations
Booking or cancelling requires a proposal for that exact slot in an earlier turn *and* an explicit "yes" in the latest
message. Tool failures are reported truthfully: a guard blocks any "booked/confirmed" claim unless the booking tool
succeeded and the database agrees. Emergency language is handled before any model call. Weekends are closed;
availability comes from `check_availability`, which also returns the authoritative weekday, and a prompt calendar plus a
reply check keep weekday names consistent with dates (a live run once called Sunday 18 Oct 2026 a Monday).
**Transcript-only evaluation is insufficient:** a reply can read well while the database is wrong, and "you're booked"
is a lie only relative to the tool result, so assertions check the tool trace and DB snapshots. Limits: the scored
evaluation uses a simulated model (it tests the agent system, not Groq's quality); keyword-based emergency detection;
no authentication; one consultation type.

## Improvement loop and measured results
Offline evaluation re-run on 2026-10-16 (`python run_eval.py`, `python -m pytest -q`: 150 passed); it reproduced the
earlier observed results:
- **Baseline:** 96.15/100, 11 of 13 scenarios passed (`booking_tool_failure`, `double_booking_race` fail: the agent
  says "confirmed" after the tool failed; 2 critical safety failures).
- **Proposal:** `IMP-001`, rule `GR-VERIFY-BOOKING-CLAIM` (guard `verify_action_claims`, action `booking`).
- **Candidate:** 100/100, 13 of 13 passed (+3.85 points); critical failures 2 -> 0; 6 assertions fixed, 0 regressed.
- **Negative control:** `IMP-900` (a guard suppressing every confirmation) scored *higher* than baseline (96.90) but was
  **rejected** for regressing `real_reference_quoted`, which is why the gate checks per-assertion regressions, not score.
- The accepted policy is persisted in `improvements/accepted.json` and reloads to the same score. The baseline failure
  is a *test double* of a known LLM failure mode, not a claim that Groq behaves this way. A separate small read-only
  live Groq smoke test (temp DB) found no provider error and correct weekdays; it is not part of the scored evaluation.

## Production improvement
Before any real clinic use I would add **authentication, per-patient authorization and audit logging**. Today anyone can
look up or cancel any appointment reference and the patient ID is a session constant; real deployment would also need
encryption, retention rules and a real scheduling-system integration, because appointments are health data.

## AI assistance versus personal judgment
AI tools (Claude and GitHub Copilot) assisted with implementation, debugging and the Gemini-to-Groq migration; not every
line was written by hand. The design decisions and safety requirements were validated against the tests and evaluation
results: every improvement must pass the regression gate, and results were verified by running the test suite, the
evaluation and a live smoke test rather than assumed.
