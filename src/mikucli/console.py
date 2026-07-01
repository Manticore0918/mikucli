from __future__ import annotations

import json
from typing import Any

from .llm import TokenUsage


class TerminalConsole:
    def progress(self, message: str) -> None:
        print(f"🤔{message}")

    def tool_request(self, name: str, arguments: dict[str, Any]) -> None:
        print(f"🔧Tools: {name} {json.dumps(arguments, ensure_ascii=False)}")

    def tool_result(self, name: str, ok: bool, content: str, diff: str = "") -> None:
        status = "ok" if ok else "failed"
        print(f"🔧Tools: {name} -> {status}")
        if content:
            print(_truncate(content))
        if diff:
            print("🔧Tools: diff")
            print(_truncate(diff, limit=8000))

    def answer(self, content: str) -> None:
        print(f"🤖Agent: {content}")

    def token_usage(self, usage: TokenUsage) -> None:
        if usage.total_tokens is None:
            print("📊Token: unavailable")
            return
        details = [f"total={usage.total_tokens}"]
        if usage.prompt_tokens is not None:
            details.append(f"prompt={usage.prompt_tokens}")
        if usage.completion_tokens is not None:
            details.append(f"completion={usage.completion_tokens}")
        print(f"📊Token: {', '.join(details)}")

    def confirm_command(self, command: str, workspace: str, reason: str) -> bool:
        print("🔧Tools: command review")
        print(f"workspace: {workspace}")
        print(f"reason: {reason}")
        print(f"command: {command}")
        answer = input("Run this command? [y/N] ").strip().lower()
        return answer in {"y", "yes"}


def _truncate(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... truncated ..."
