"""Live backend using the official ``groq`` SDK chat-completions interface.

Tool calling is driven manually so every tool call goes through ``ToolExecutor`` -
the single place that injects the patient ID, enforces the confirmation gate and
records the trace.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from groq import Groq

from .agent import ToolExecutor
from .errors import ProviderError
from .state import ConversationState
from .tools import MODEL_TOOL_SPECS

MAX_TOOL_ROUNDS = 6
REQUEST_TIMEOUT_SECONDS = 30

_JSON_TYPE = {
    "STRING": "string",
    "NUMBER": "number",
    "INTEGER": "integer",
    "BOOLEAN": "boolean",
    "ARRAY": "array",
    "OBJECT": "object",
}


def build_tool_declarations() -> list[dict[str, Any]]:
    declarations: list[dict[str, Any]] = []
    for spec in MODEL_TOOL_SPECS:
        props: dict[str, Any] = {}
        for name, p in spec["properties"].items():
            prop: dict[str, Any] = {"type": _JSON_TYPE[p["type"]]}
            if "description" in p:
                prop["description"] = p["description"]
            if "enum" in p:
                prop["enum"] = p["enum"]
            props[name] = prop
        declarations.append(
            {
                "type": "function",
                "function": {
                    "name": spec["name"],
                    "description": spec["description"],
                    "parameters": {"type": "object", "properties": props, "required": spec["required"]},
                },
            }
        )
    return declarations


def make_client(api_key: str) -> Groq:
    return Groq(api_key=api_key, timeout=REQUEST_TIMEOUT_SECONDS)


def _sanitize(text: str, secret: Optional[str]) -> str:
    if secret:
        text = text.replace(secret, "[redacted]")
    return text[:200]


class GroqBackend:
    name = "groq"

    def __init__(self, client: Any, model: str, api_key: Optional[str] = None, temperature: float = 0.2):
        self.client = client
        self.model = model
        self._api_key = api_key  # only used to redact it from error text
        self.temperature = temperature
        self._tools = build_tool_declarations()

    def _history_messages(self, state: ConversationState, user_message: str, system_prompt: str) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        for item in state.history[:-1]:  # current user message is appended explicitly below
            role = "user" if item["role"] == "user" else "assistant"
            messages.append({"role": role, "content": item["text"]})
        messages.append({"role": "user", "content": user_message})
        return messages

    def generate_reply(
        self, user_message: str, state: ConversationState, executor: ToolExecutor, system_prompt: str
    ) -> str:
        messages = self._history_messages(state, user_message, system_prompt)
        for _ in range(MAX_TOOL_ROUNDS):
            response = self._generate(messages)
            if not getattr(response, "choices", None):
                raise ProviderError("the model returned an empty reply", "empty_response")

            message = response.choices[0].message
            tool_calls = list(getattr(message, "tool_calls", None) or [])
            if not tool_calls:
                text = (getattr(message, "content", None) or "").strip()
                if not text:
                    raise ProviderError("the model returned an empty reply", "empty_response")
                return text

            assistant_call = {"role": "assistant", "content": getattr(message, "content", None) or "", "tool_calls": []}
            for call in tool_calls:
                assistant_call["tool_calls"].append(
                    {
                        "id": getattr(call, "id", "") or "",
                        "type": "function",
                        "function": {
                            "name": getattr(getattr(call, "function", None), "name", "") or "",
                            "arguments": getattr(getattr(call, "function", None), "arguments", "") or "{}",
                        },
                    }
                )
            messages.append(assistant_call)

            for i, call in enumerate(tool_calls, start=1):
                function = getattr(call, "function", None)
                name = getattr(function, "name", "") or ""
                raw_args = getattr(function, "arguments", "") or "{}"
                try:
                    parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args or {})
                except (json.JSONDecodeError, TypeError, ValueError):
                    parsed_args = {}
                result = executor.call(name, parsed_args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": getattr(call, "id", None) or f"toolcall-{i}",
                        "name": name,
                        "content": json.dumps({"result": result.model_dump()}, ensure_ascii=True),
                    }
                )
        raise ProviderError("the model made too many tool calls in one turn", "tool_loop")

    def _generate(self, messages: list[dict[str, Any]]) -> Any:
        try:
            return self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self._tools,
                tool_choice="auto",
                temperature=self.temperature,
            )
        except Exception as exc:
            name = type(exc).__name__.lower()
            status = getattr(exc, "status_code", None)
            if status == 429 or "ratelimit" in name:
                raise ProviderError("the API quota or rate limit was reached; wait a bit and try again", "quota") from None
            if status in (401, 403) or "authentication" in name:
                raise ProviderError("the API key was rejected; check GROQ_API_KEY in .env", "auth") from None
            if status == 404:
                raise ProviderError("the configured model was not found on Groq", "model_not_found") from None
            if isinstance(status, int) and status >= 500:
                raise ProviderError("the Groq service is temporarily unavailable", "server_error") from None
            if "timeout" in name:
                raise ProviderError(f"timeout problem ({type(exc).__name__}: {_sanitize(str(exc), self._api_key)})", "timeout") from None
            if "connection" in name or "connect" in name:
                raise ProviderError(f"network problem ({type(exc).__name__}: {_sanitize(str(exc), self._api_key)})", "network") from None
            raise ProviderError(f"API error ({type(exc).__name__})", "api_error") from None
