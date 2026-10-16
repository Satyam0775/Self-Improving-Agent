"""Terminal chat interface for the appointment agent."""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

from .agent import AppointmentAgent
from .config import IMPROVEMENTS_DIR, SETUP_HELP, load_settings
from .database import Database
from .policy import load_active_policy
from .tools import AppointmentTools

EXIT_WORDS = {"exit", "quit", "/exit", "/quit", "bye"}

BANNER = """\
============================================================
 Appointment Assistant  -  SIMULATED CLINIC DEMO
 Appointments here are fake and stored only in a local
 SQLite file. This is not a real clinic or hospital.
 Do not enter real personal or medical information.
 Type 'exit' or 'quit' to leave.
============================================================"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Chat with the simulated appointment scheduling agent.")
    p.add_argument("--simulated", action="store_true", help="use the offline rule-based model instead of Groq (no API key needed)")
    p.add_argument("--patient-id", default="PAT-1001", help="synthetic patient ID (default: PAT-1001)")
    p.add_argument("--db", type=Path, default=None, help="SQLite file (default: data/appointments.db)")
    p.add_argument("--reset-db", action="store_true", help="delete the local demo database before starting")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings()

    if not args.simulated and not settings.has_api_key:
        print(SETUP_HELP)
        return 2

    db_path = args.db or settings.db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if args.reset_db and db_path.exists():
        db_path.unlink()
    today = dt.date.today()
    db = Database(db_path)
    db.init_schema()
    db.seed_schedule(today)

    policy = load_active_policy(IMPROVEMENTS_DIR)
    if args.simulated:
        from .simulated_model import SimulatedModel

        backend = SimulatedModel(today)
        backend_name = "offline simulated model (not Groq)"
    else:
        from .groq_backend import GroqBackend, make_client

        backend = GroqBackend(make_client(settings.groq_api_key), settings.groq_model, settings.groq_api_key)
        backend_name = f"Groq model '{settings.groq_model}'"

    try:
        agent = AppointmentAgent(AppointmentTools(db), policy, backend, today, patient_id=args.patient_id)
    except ValueError as exc:
        print(f"Cannot start: {exc}")
        return 1

    print(BANNER)
    print(f"Backend: {backend_name} | policy: {policy.name} | patient: {args.patient_id}")
    print("Assistant: Hello! I can help you book, view or cancel an appointment. How can I help?\n")

    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return 0
        if not text:
            continue
        if text.lower() in EXIT_WORDS:
            print("Assistant: Goodbye. Remember, this was a simulated demo.")
            return 0
        try:
            result = agent.handle_message(text)
        except Exception as exc:  # last-resort safety net; never show a raw traceback to the patient
            print(f"Assistant: Sorry, something went wrong ({type(exc).__name__}). Nothing was changed by this message.\n")
            continue
        print(f"Assistant: {result.reply}\n")
        if result.provider_error:
            print(f"  [diagnostic: {result.provider_error}]\n")


if __name__ == "__main__":
    sys.exit(main())
