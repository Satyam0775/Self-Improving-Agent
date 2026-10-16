"""Configuration: environment variables (via python-dotenv) and project paths."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "openai/gpt-oss-120b"  # internal default for Groq live chat
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "appointments.db"
IMPROVEMENTS_DIR = PROJECT_ROOT / "improvements"
REPORTS_DIR = PROJECT_ROOT / "reports"
SCENARIOS_FILE = PROJECT_ROOT / "evals" / "scenarios.yaml"

_PLACEHOLDER_PREFIXES = ("replace_with", "your_", "<")

SETUP_HELP = """\
No Groq API key is configured, so the live chat cannot start.

To set it up:
  1. Copy the example file:   copy .env.example .env        (PowerShell: Copy-Item .env.example .env)
  2. Open ".env" in VS Code and paste your own key after GROQ_API_KEY=
     (create a key at https://console.groq.com/keys).
  3. Run again:              python run_agent.py

Want to try the chat without a key? Run:  python run_agent.py --simulated
(this uses the offline rule-based model, not Groq).
The deterministic evaluation never needs a key:  python run_eval.py
"""


def _is_placeholder(value: str) -> bool:
    v = value.strip().lower()
    return not v or v.startswith(_PLACEHOLDER_PREFIXES)


@dataclass(frozen=True)
class Settings:
    groq_api_key: str
    groq_model: str
    db_path: Path

    @property
    def has_api_key(self) -> bool:
        return not _is_placeholder(self.groq_api_key)

    def __repr__(self) -> str:  # never leak the key through logs / tracebacks
        return f"Settings(groq_model={self.groq_model!r}, has_api_key={self.has_api_key}, db_path={str(self.db_path)!r})"


def load_settings(env_file: Path | None = None) -> Settings:
    load_dotenv(env_file or PROJECT_ROOT / ".env", override=False)
    return Settings(
        groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
        groq_model=DEFAULT_MODEL,
        db_path=Path(os.getenv("APPOINTMENTS_DB", str(DEFAULT_DB_PATH))),
    )
