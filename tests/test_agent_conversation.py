"""Multi-turn conversation behaviour (offline simulated model)."""

from app.nlu import parse_date, parse_time


def test_state_is_preserved_across_turns_and_booking_lands_on_remembered_slot(make_agent, db, slot):
    agent = make_agent()
    agent.handle_message("Book me something on Wednesday.")
    assert agent.state.date == "2026-03-04"
    agent.handle_message("General consultation, mornings please.")
    assert agent.state.date == "2026-03-04" and agent.state.time_preference == "morning"
    confirm = agent.handle_message("The 10 am one.")
    assert "Wednesday 4 March 2026" in confirm.reply and "10:00 AM" in confirm.reply
    assert agent.state.pending_action and agent.state.pending_action.target_id == slot(2, "10:00")
    done = agent.handle_message("Yes.")
    assert "APT-00001" in done.reply
    appt = db.snapshot()["appointments"]
    assert appt == [{"appointment_id": "APT-00001", "patient_id": "PAT-1001", "slot_id": slot(2, "10:00"), "status": "booked"}]


def test_vague_request_gets_a_clarifying_question_not_a_guess(make_agent, db):
    agent = make_agent()
    res = agent.handle_message("I need an appointment sometime soon")
    assert "?" in res.reply and "specific day" in res.reply
    assert res.tool_trace == [] and db.snapshot()["appointments"] == []


def test_declining_confirmation_creates_nothing(make_agent, db):
    agent = make_agent()
    agent.handle_message("general consultation tomorrow at 4 PM please")
    res = agent.handle_message("No thanks")
    assert "not booked" in res.reply and agent.state.pending_action is None
    assert db.snapshot()["appointments"] == []


def test_changing_details_at_confirmation_restarts_the_proposal(make_agent, db, slot):
    agent = make_agent()
    agent.handle_message("general consultation tomorrow at 4 PM please")
    res = agent.handle_message("No, make it 2 PM instead")
    assert "2:00 PM" in res.reply and agent.state.pending_action.target_id == slot(1, "14:00")
    assert db.snapshot()["appointments"] == []


def test_taken_slot_is_reported_unavailable_with_alternatives(make_agent, tools, slot):
    tools.book_appointment("PAT-2001", slot(1, "16:00"), True)
    res = make_agent().handle_message("general consultation tomorrow at 4 PM")
    assert "not available" in res.reply and "2:00 PM" in res.reply


def test_view_and_cancel_flow(make_agent, tools, db, slot):
    tools.book_appointment("PAT-1001", slot(1, "16:00"), True)
    agent = make_agent()
    view = agent.handle_message("What are the details of APT-00001?")
    assert "4:00 PM" in view.reply and "booked" in view.reply
    ask = agent.handle_message("please cancel APT-00001")
    assert "?" in ask.reply and db.get_appointment("APT-00001")["status"] == "booked"
    done = agent.handle_message("yes")
    assert "cancelled" in done.reply and db.get_appointment("APT-00001")["status"] == "cancelled"


def test_cancel_unknown_reference_changes_nothing(make_agent, tools, db, slot):
    tools.book_appointment("PAT-1001", slot(1, "16:00"), True)
    before = db.snapshot()
    res = make_agent().handle_message("cancel APT-99999")
    assert "couldn't find" in res.reply and db.snapshot() == before


def test_date_and_time_parsing(today):
    assert parse_date("tomorrow", today).date == "2026-03-03"
    assert parse_date("next Friday", today).date == "2026-03-06"
    assert parse_date("March 9th", today).date == "2026-03-09"
    assert parse_date("sometime next week", today).ambiguous
    assert parse_time("at 4 PM") == "16:00" and parse_time("10:30am") == "10:30" and parse_time("4") == "16:00"
    assert parse_time("APT-00012 on 2026-03-04") is None
