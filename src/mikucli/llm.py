from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


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


class BigModelClient:
    def __init__(self, api_key: str, base_url: str) -> None:
        self.api_key = api_key
        self.base_url = base_url

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
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"BigModel API error {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"BigModel API request failed: {exc.reason}") from exc

        raw = json.loads(body)
        message = raw["choices"][0]["message"]
        return AssistantMessage(
            content=message.get("content") or "",
            tool_calls=_parse_tool_calls(message.get("tool_calls") or []),
            raw=raw,
            token_usage=_parse_token_usage(raw.get("usage") or {}),
        )


def _parse_tool_calls(raw_calls: list[dict[str, Any]]) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for index, raw_call in enumerate(raw_calls):
        function = raw_call.get("function") or {}
        raw_arguments = function.get("arguments") or "{}"
        if isinstance(raw_arguments, str):
            arguments = json.loads(raw_arguments)
        elif isinstance(raw_arguments, dict):
            arguments = raw_arguments
        else:
            arguments = {}
        calls.append(
            ToolCall(
                id=str(raw_call.get("id") or f"call_{index}"),
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
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
