from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import httpx

from .http_requests import ModelResponseError, RequestPolicy, request_json
from .run_control import current_budget


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class AssistantMessage:
    content: str
    tool_calls: list[ToolCall]
    raw: dict[str, Any]
    token_usage: TokenUsage
    request_attempts: int = 1


class ModelClient:
    """Client for OpenAI-compatible Chat Completions endpoints."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        request_policy: RequestPolicy | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self._transport = transport
        self.request_policy = request_policy or RequestPolicy()

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        stream: bool = False,
    ) -> AssistantMessage:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if stream:
            raise ValueError("Streaming responses are not supported by this client.")
        budget = current_budget()
        if budget is not None and budget.limits.max_tokens is not None:
            budget.check()
            payload["max_tokens"] = min(4096, max(1, budget.limits.max_tokens - budget.tokens))
        raw, attempts = asyncio.run(request_json(
            self.base_url, policy=self.request_policy, payload=payload,
            headers={"Authorization": f"Bearer {self.api_key}"}, transport=self._transport,
        ))
        usage = _parse_token_usage(raw.get("usage") or {}) if isinstance(raw, dict) and isinstance(raw.get("usage", {}), dict) else TokenUsage()
        try:
            choice = raw["choices"][0]
            if choice.get("finish_reason") in {"length", "content_filter"}:
                raise ModelResponseError("Provider response was incomplete or filtered.")
            message = choice["message"]
            content = message.get("content") or ""
            if not isinstance(content, str):
                raise ModelResponseError("Provider message content must be text.")
            calls = _parse_tool_calls(message.get("tool_calls") or [])
            if not content.strip() and not calls:
                raise ModelResponseError("Provider returned an empty answer.")
            usage = _parse_token_usage(raw.get("usage") or {})
        except (ModelResponseError, KeyError, IndexError, TypeError, AttributeError, ValueError) as exc:
            error = exc if isinstance(exc, ModelResponseError) else ModelResponseError("Provider returned a malformed chat response.")
            error.token_usage = usage
            error.request_attempts = attempts
            raise error from (exc if error is not exc else None)
        return AssistantMessage(
            content=content,
            tool_calls=calls,
            raw=raw,
            token_usage=usage,
            request_attempts=attempts,
        )


# Backward-compatible import for integrations built against earlier mikucli releases.
BigModelClient = ModelClient


def _parse_tool_calls(raw_calls: list[dict[str, Any]]) -> list[ToolCall]:
    if not isinstance(raw_calls, list):
        raise ModelResponseError("Provider tool calls must be a list.")
    if len(raw_calls) > 128:
        raise ModelResponseError("Provider exceeded the limit of 128 tool calls per response.")
    calls: list[ToolCall] = []
    identifiers: set[str] = set()
    for index, raw_call in enumerate(raw_calls):
        function = raw_call.get("function") or {}
        raw_arguments = function.get("arguments") or "{}"
        if isinstance(raw_arguments, str):
            arguments = json.loads(raw_arguments)
        elif isinstance(raw_arguments, dict):
            arguments = raw_arguments
        else:
            raise ModelResponseError("Provider tool arguments must be a JSON object.")
        if not isinstance(arguments, dict) or not isinstance(function.get("name"), str) or not function["name"]:
            raise ModelResponseError("Provider tool call has invalid arguments or name.")
        identifier = str(raw_call.get("id") or f"call_{index}")
        if identifier in identifiers:
            raise ModelResponseError("Provider returned duplicate tool-call identifiers.")
        identifiers.add(identifier)
        calls.append(
            ToolCall(
                id=identifier,
                name=str(function.get("name") or ""),
                arguments=arguments,
            )
        )
    return calls


def _parse_token_usage(raw_usage: dict[str, Any]) -> TokenUsage:
    return TokenUsage(
        prompt_tokens=_int_or_none(raw_usage.get("prompt_tokens")),
        completion_tokens=_int_or_none(raw_usage.get("completion_tokens")),
        total_tokens=_int_or_none(raw_usage.get("total_tokens")),
    )


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None
