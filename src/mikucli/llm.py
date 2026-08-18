from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import httpx

from .cancellation import StopCallback, StopRequested, current_stop_requested


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


class ModelClient:
    """Client for OpenAI-compatible Chat Completions endpoints."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self._transport = transport

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
        raw = asyncio.run(self._post(payload))
        message = raw["choices"][0]["message"]
        return AssistantMessage(
            content=message.get("content") or "",
            tool_calls=_parse_tool_calls(message.get("tool_calls") or []),
            raw=raw,
            token_usage=_parse_token_usage(raw.get("usage") or {}),
        )

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        stop_requested = current_stop_requested()
        if stop_requested is not None and stop_requested():
            raise StopRequested("stopped by user")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            async with httpx.AsyncClient(
                transport=self._transport,
                timeout=httpx.Timeout(120.0),
            ) as client:
                request_task = asyncio.create_task(
                    client.post(self.base_url, json=payload, headers=headers)
                )
                if stop_requested is None:
                    response = await request_task
                else:
                    stop_task = asyncio.create_task(_wait_for_stop(stop_requested))
                    done, _ = await asyncio.wait(
                        {request_task, stop_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if stop_task in done and stop_requested():
                        request_task.cancel()
                        await asyncio.gather(request_task, return_exceptions=True)
                        raise StopRequested("stopped by user")
                    stop_task.cancel()
                    await asyncio.gather(stop_task, return_exceptions=True)
                    response = await request_task
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"Model API error {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.RequestError as exc:
            raise RuntimeError(f"Model API request failed: {exc}") from exc
        return response.json()


# Backward-compatible import for integrations built against earlier mikucli releases.
BigModelClient = ModelClient


async def _wait_for_stop(stop_requested: StopCallback) -> None:
    while not stop_requested():
        await asyncio.sleep(0.05)


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
