"""Deterministic patient-facing message templates."""

from __future__ import annotations

DISCLOSURE = (
    "Please note: this is a simulated scheduling demo, not a real clinic, "
    "and no real medical care is arranged here."
)

EMERGENCY_MESSAGE = (
    "I'm concerned about what you've described. These symptoms may need urgent medical attention, "
    "and I can't assess, diagnose or treat them. Please call your local emergency number now "
    "(for example 911 in the US or 112 in many other countries) or go to the nearest emergency "
    "department. If someone is with you, ask them to help you get care. If you are thinking about "
    "harming yourself, please contact emergency services or a crisis line right away. "
    "I am a simulated scheduling assistant, so I will not continue with routine booking right now."
)

EMERGENCY_REMINDER = (
    "Your safety comes first. Please contact your local emergency number or go to the nearest "
    "emergency department before arranging a routine appointment. I am not able to assess your "
    "symptoms. If this is not an emergency, tell me so and I can continue with scheduling."
)

EMERGENCY_CLEARED = (
    "Thank you for letting me know. Since this is not an emergency, I can help with routine "
    "scheduling. Would you like to book, view or cancel an appointment?"
)


def booking_confirmed(label: str, reference: str) -> str:
    return (
        f"Your appointment is confirmed for {label}. Your reference is {reference}. "
        "This is a simulated booking."
    )


def booking_failed(detail: str) -> str:
    return (
        "I'm sorry, but I was not able to complete that booking, so no appointment has been created. "
        f"{detail} Would you like me to try again or look at other available times?"
    ).replace("  ", " ")


def cancellation_confirmed(reference: str, label: str) -> str:
    return f"Appointment {reference} ({label}) has been cancelled."


def cancellation_failed(detail: str) -> str:
    return (
        "I'm sorry, but I was not able to cancel that appointment, so nothing has been changed. "
        f"{detail} Would you like to double-check the reference and try again?"
    ).replace("  ", " ")


def provider_unavailable(detail: str) -> str:
    return (
        "I'm sorry, I'm having trouble reaching my language service right now "
        f"({detail}). No changes were made to your appointments in this step. "
        "Please try again in a moment."
    )
