"""Guardrail test: active app code should not depend on Gemini after Groq migration."""

from pathlib import Path


def test_no_active_gemini_backend_or_env_in_runtime_paths():
    root = Path(__file__).resolve().parent.parent
    active_files = [
        root / "app" / "config.py",
        root / "app" / "main.py",
        root / "run_eval.py",
        root / "requirements.txt",
        root / ".env.example",
    ]
    banned = ("GEMINI_API_KEY", "GEMINI_MODEL", "app.gemini_backend", "google-genai")
    for path in active_files:
        text = path.read_text(encoding="utf-8")
        for needle in banned:
            assert needle not in text, f"{needle} still present in {path.name}"

