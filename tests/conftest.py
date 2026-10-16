"""Shared fixtures. Every test uses its own temporary database; no network or API key."""

import datetime as dt

import pytest

from app.agent import AppointmentAgent
from app.database import Database, make_slot_id
from app.policy import BASELINE_POLICY
from app.simulated_model import SimulatedModel
from app.tools import AppointmentTools

TODAY = dt.date(2026, 3, 2)  # Monday


@pytest.fixture
def today():
    return TODAY


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "test.db")
    database.init_schema()
    database.seed_schedule(TODAY)
    return database


@pytest.fixture
def tools(db):
    return AppointmentTools(db)


@pytest.fixture
def slot():
    """slot_id helper: slot(day_offset, 'HH:MM')."""
    return lambda offset, hhmm: make_slot_id(TODAY + dt.timedelta(days=offset), hhmm)


@pytest.fixture
def make_agent(db):
    def factory(policy=BASELINE_POLICY, backend=None, fail_tools=None, optimistic=True):
        tools = AppointmentTools(db, fail_tools=fail_tools)
        backend = backend or SimulatedModel(TODAY, optimistic_booking_claims=optimistic)
        return AppointmentAgent(tools, policy, backend, TODAY, patient_id="PAT-1001")

    return factory
