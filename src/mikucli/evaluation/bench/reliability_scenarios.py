"""Local HTTP transport scripts driving real agent sessions and tool execution."""

from __future__ import annotations

import asyncio
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Thread, Timer

import httpx

from mikucli.http_requests import RequestPolicy
from mikucli.llm import ModelClient
from mikucli.run_control import RunLimits

from .reliability_tasks import CONTRACTS


@dataclass
class Scenario:
    name: str
    workspace: Path
    expected_attempts: int
    attempts: int = 0
    cancelled_requests: int = 0
    errors: list[str] = field(default_factory=list)
    stop: Event = field(default_factory=Event)
    timers: list[Timer | Thread] = field(default_factory=list)
    limits: RunLimits = field(
        default_factory=lambda: RunLimits(max_duration_seconds=10)
    )
    tools: object | None = None
    client: ModelClient | None = None

    def later_stop(self) -> None:
        timer = Timer(0.08, self.stop.set)
        timer.start()
        self.timers.append(timer)

    def stop_running_tool(self) -> None:
        def wait_for_start():
            deadline = time.monotonic() + 5
            while (
                not (self.workspace / "started").exists()
                and time.monotonic() < deadline
            ):
                if self.stop.wait(0.01):
                    return
            self.stop.set()

        thread = Thread(target=wait_for_start, daemon=True)
        thread.start()
        self.timers.append(thread)

    def verify(self) -> None:
        if (
            self.name in {"cancel_request", "duration_limit"}
            and self.cancelled_requests != 1
        ):
            self.errors.append("in-flight HTTP request was not canceled")
        if self.name == "provider_request_timeout" and self.cancelled_requests != 2:
            self.errors.append("timed-out HTTP requests were not canceled before retry")
        if self.name == "cancel_tool" and not (self.workspace / "started").exists():
            self.errors.append(
                "tool cancellation was not exercised after process startup"
            )

    def close(self) -> None:
        self.stop.set()
        for timer in self.timers:
            if isinstance(timer, Timer):
                timer.cancel()
            timer.join(timeout=1)


def response(content="done", *, tool=None, tokens=3, usage=True):
    message = {"content": content}
    if tool is not None:
        name, arguments = tool
        message["tool_calls"] = [
            {
                "id": "contract_call",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ]
    payload = {
        "choices": [
            {"message": message, "finish_reason": "tool_calls" if tool else "stop"}
        ]
    }
    if usage:
        payload["usage"] = {
            "prompt_tokens": 2,
            "completion_tokens": tokens - 2,
            "total_tokens": tokens,
        }
    return httpx.Response(200, json=payload)


def make_scenario(name: str, workspace: Path) -> Scenario:
    attempts = next(count for ident, _, _, count in CONTRACTS if ident == name)
    scenario = Scenario(name, workspace, attempts)
    options = {"max_retry_delay": 0, "max_retries": 0, "timeout_seconds": 1}
    if name == "provider_retry_recovery":
        options["max_retries"] = 3
    elif name == "provider_exhaustion":
        options["max_retries"] = 2
    elif name == "provider_request_timeout":
        options.update(max_retries=1, timeout_seconds=0.04)
    elif name == "cancel_backoff":
        options.update(max_retries=2, max_retry_delay=10)
    if name in {"token_limit", "usage_unavailable", "team_shared_budget"}:
        scenario.limits = RunLimits(max_tokens=8 if name.startswith("team_") else 3)
    elif name == "spend_limit":
        scenario.limits = RunLimits(
            max_spend=0.000002,
            prompt_price_per_million=1,
            completion_price_per_million=1,
        )
    elif name in {"duration_limit", "mcp_queued_cancellation"}:
        scenario.limits = RunLimits(max_duration_seconds=0.1)
    elif name == "output_limit":
        scenario.limits = RunLimits(max_output_bytes=256)

    async def handler(request):
        scenario.attempts += 1
        turn = scenario.attempts
        if name == "provider_retry_recovery" and turn < 4:
            if turn == 3:
                raise httpx.ConnectError("fixture disconnected")
            return httpx.Response(
                503 if turn == 1 else 429, headers={"Retry-After": "0"}
            )
        if (
            name == "provider_exhaustion"
            or (name == "team_worker_failure" and turn == 2)
            or (name == "team_reviewer_failure" and turn == 3)
        ):
            return httpx.Response(503)
        if name == "provider_invalid_response":
            return httpx.Response(200, json={"choices": []})
        if name == "cancel_backoff":
            scenario.later_stop()
            return httpx.Response(429, headers={"Retry-After": "5"})
        if name in {"cancel_request", "duration_limit", "provider_request_timeout"}:
            if name == "cancel_request":
                scenario.stop.set()
            try:
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                scenario.cancelled_requests += 1
                raise
        if name.startswith("durable_"):
            from .reliability_state import exercise_memory

            scenario.errors.extend(exercise_memory(name, workspace))
        if name.startswith("team_") and turn == 1:
            return response(
                json.dumps(
                    {"steps": [{"id": "one", "task": "perform the fixture work"}]}
                ),
                tokens=5,
            )
        if name == "team_shared_budget":
            return response(tokens=4)
        if (
            name
            in {
                "token_limit",
                "spend_limit",
                "usage_unavailable",
                "approval_denial",
                "cancel_approval",
            }
            and turn == 1
        ):
            return response(
                "",
                tool=(
                    "write_file",
                    {"path": "late.txt", "content": "must never execute"},
                ),
                usage=name != "usage_unavailable",
            )
        if name in {"output_limit", "cancel_tool"}:
            script = (
                "print('x' * 20000)"
                if name == "output_limit"
                else (
                    "from pathlib import Path; import time; Path('started').write_text('ready'); "
                    "time.sleep(5); Path('late.txt').write_text('late')"
                )
            )
            arguments = [sys.executable, "-c", script]
            command = (
                subprocess.list2cmdline(arguments)
                if sys.platform == "win32"
                else shlex.join(arguments)
            )
            return response(
                "",
                tool=(
                    "run_shell",
                    {"command": command, "reason": "offline reliability probe"},
                ),
            )
        if name == "mcp_queued_cancellation":
            return response("", tool=("remote_tool", {}))
        return response()

    scenario.client = ModelClient(
        "offline-fixture",
        "https://offline.invalid/chat",
        transport=httpx.MockTransport(handler),
        request_policy=RequestPolicy(**options),
    )
    return scenario
