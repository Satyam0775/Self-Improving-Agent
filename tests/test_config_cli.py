"""Configuration and CLI behaviour."""

import pytest

from app import config
from app.config import DEFAULT_MODEL, Settings, load_settings
from app.main import main


def test_default_model_is_current_groq_replacement():
    assert DEFAULT_MODEL == "openai/gpt-oss-120b"


def test_placeholder_or_missing_key_is_not_a_key(monkeypatch, tmp_path):
    for value in ["", "replace_with_your_own_groq_api_key", "your_groq_api_key_here"]:
        monkeypatch.setenv("GROQ_API_KEY", value)
        assert not load_settings(tmp_path / "none.env").has_api_key


def test_real_looking_key_and_default_model_are_read(monkeypatch, tmp_path):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key-for-tests")
    s = load_settings(tmp_path / "none.env")
    assert s.has_api_key and s.groq_model == DEFAULT_MODEL


def test_model_is_not_loaded_from_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("GROQ_MODEL", "not-used-by-config")
    assert load_settings(tmp_path / "none.env").groq_model == DEFAULT_MODEL


def test_dotenv_file_is_loaded(monkeypatch, tmp_path):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("GROQ_API_KEY=fake-key-from-file\n", encoding="utf-8")
    assert load_settings(env).groq_api_key == "fake-key-from-file"
    monkeypatch.delenv("GROQ_API_KEY", raising=False)


def test_settings_repr_never_contains_the_key():
    s = Settings(groq_api_key="fake-secret", groq_model="m", db_path=config.DEFAULT_DB_PATH)
    assert "fake-secret" not in repr(s) and "fake-secret" not in str(s)


def test_live_chat_without_key_prints_setup_help_not_a_traceback(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("GROQ_API_KEY", "your_groq_api_key_here")
    code = main(["--db", str(tmp_path / "x.db")])
    out = capsys.readouterr().out
    assert code == 2 and ".env.example" in out and "GROQ_API_KEY" in out and "Traceback" not in out
    assert not (tmp_path / "x.db").exists()  # nothing started


def test_simulated_chat_session_end_to_end(monkeypatch, capsys, tmp_path):
    inputs = iter(["hello", "", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    assert main(["--simulated", "--db", str(tmp_path / "chat.db")]) == 0
    out = capsys.readouterr().out
    assert "SIMULATED CLINIC DEMO" in out and "Goodbye" in out


def test_ctrl_d_exits_cleanly(monkeypatch, capsys, tmp_path):
    def raise_eof(prompt=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", raise_eof)
    assert main(["--simulated", "--db", str(tmp_path / "chat.db")]) == 0
