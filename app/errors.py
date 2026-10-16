"""Shared exception types."""


class ProviderError(Exception):
    """The LLM provider failed (quota, timeout, auth, network, ...).

    ``user_message`` is safe to show to a patient and never contains secrets.
    """

    def __init__(self, user_message: str, kind: str = "provider_error"):
        super().__init__(user_message)
        self.user_message = user_message
        self.kind = kind


class PolicyError(Exception):
    """A policy or improvement record is invalid or unsafe."""
