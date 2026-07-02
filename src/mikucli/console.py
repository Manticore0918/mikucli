from __future__ import annotations

import json
from typing import Any

from .llm import TokenUsage
from .tools import ToolApprovalRequest


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

    def confirm_tool(self, request: ToolApprovalRequest) -> bool:
        print("🔧Tools: tool approval")
        print(f"risk: {request.risk_level.value}")
        print(f"workspace: {request.workspace}")
        print(request.summary)
        if request.details:
            print(_truncate(request.details, limit=8000))
        prompt = "Apply this file change? [y/N] " if request.tool_name == "write_file" else "Run this tool? [y/N] "
        answer = input(prompt).strip().lower()
        return answer in {"y", "yes"}


def _truncate(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... truncated ..."
