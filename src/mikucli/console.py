from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable
from typing import Any, Literal

from .llm import TokenUsage
from .skills import SkillEntry
from .tools import ToolApprovalRequest


LanguageCode = Literal["eng", "chn"]


def _ui(language: LanguageCode, english: str, chinese: str) -> str:
    return chinese if language == "chn" else english


class TerminalConsole:
    def __init__(self, language: LanguageCode = "eng") -> None:
        self.language = language
        self._approval_handler: Callable[[ToolApprovalRequest], bool] | None = None

    def set_language(self, language: LanguageCode) -> None:
        self.language = language

    def prompt_label(self) -> str:
        return _ui(self.language, "You: ", "你: ")

    def progress(self, message: str) -> None:
        print(f"🤔{self._progress_message(message)}")

    def tool_request(self, name: str, arguments: dict[str, Any]) -> None:
        label = _ui(self.language, "Tools", "工具")
        print(f"🔧{label}: {name} {json.dumps(arguments, ensure_ascii=False)}")

    def tool_result(self, name: str, ok: bool, content: str, diff: str = "") -> None:
        label = _ui(self.language, "Tools", "工具")
        status = _ui(self.language, "ok", "成功") if ok else _ui(self.language, "failed", "失败")
        print(f"🔧{label}: {name} -> {status}")
        if content:
            print(_truncate(content))
        if diff:
            print(f"🔧{label}: {_ui(self.language, 'diff', '差异')}")
            print(_truncate(diff, limit=8000))

    def answer(self, content: str) -> None:
        label = _ui(self.language, "Agent", "智能体")
        print(f"🤖{label}: {content}")

    def token_usage(self, usage: TokenUsage) -> None:
        label = _ui(self.language, "Token", "Token")
        if usage.total_tokens is None:
            print(f"📊{label}: {_ui(self.language, 'unavailable', '不可用')}")
            return
        details = [f"total={usage.total_tokens}"]
        if usage.prompt_tokens is not None:
            details.append(f"prompt={usage.prompt_tokens}")
        if usage.completion_tokens is not None:
            details.append(f"completion={usage.completion_tokens}")
        print(f"📊{label}: {', '.join(details)}")

    def confirm_tool(self, request: ToolApprovalRequest) -> bool:
        label = _ui(self.language, "Tools", "工具")
        print(f"🔧{label}: {_ui(self.language, 'tool approval', '工具审批')}")
        print(f"{_ui(self.language, 'risk', '风险')}: {request.risk_level.value}")
        print(f"{_ui(self.language, 'workspace', '工作区')}: {request.workspace}")
        print(request.summary)
        if request.details:
            print(_truncate(request.details, limit=8000))
        if request.tool_name == "write_file":
            prompt = _ui(self.language, "Apply this file change? [y/N] ", "应用此文件更改？[y/N] ")
        elif request.tool_name == "read_file":
            prompt = _ui(self.language, "Read this sensitive file? [y/N] ", "读取此敏感文件？[y/N] ")
        else:
            prompt = _ui(self.language, "Run this tool? [y/N] ", "运行此工具？[y/N] ")
        if self._approval_handler is not None:
            print(prompt)
            return self._approval_handler(request)
        answer = input(prompt).strip().lower()
        return answer in {"y", "yes"}

    def set_approval_handler(self, handler: Callable[[ToolApprovalRequest], bool] | None) -> None:
        self._approval_handler = handler

    def language_changed(self) -> None:
        print(_ui(self.language, "mikucli: language switched to English.", "mikucli：界面语言已切换为中文。"))

    def error(self, error: Exception) -> str:
        localized = getattr(error, "localized", None)
        if callable(localized):
            return f"mikucli: {localized(self.language)}" if self.language == "eng" else f"mikucli：{localized(self.language)}"
        return _ui(self.language, f"mikucli: {error}", f"mikucli：{error}")

    def log_path(self, path: Any) -> None:
        print(_ui(self.language, f"[log] {path}", f"[日志] {path}"))

    def interactive_intro(self) -> None:
        print(
            _ui(
                self.language,
                "mikucli interactive session. Type /stop to stop the current process; /skills, /dashboard, /team, /mcp, /eval run, /eval run-back, /eval stop, /lang-chn, /lang-eng, and /exit are also available.",
                "mikucli 交互会话。输入 /stop 停止当前进程；还可使用 /skills、/dashboard、/team、/mcp、/eval run、/eval run-back、/eval stop、/lang-chn、/lang-eng 和 /exit。",
            )
        )

    def print_skills(self, entries: list[SkillEntry]) -> None:
        if not entries:
            print(_ui(self.language, "[skills] no Skills found.", "[skills] 未找到 Skill。"))
            return
        print(_ui(self.language, "[skills] available Skills", "[skills] 可用 Skill"))
        for entry in entries:
            source = _ui(self.language, entry.scope.value, "工作区" if entry.scope.value == "workspace" else "用户")
            override = (
                _ui(self.language, "; overrides user", "；覆盖用户 Skill")
                if entry.shadows_user
                else ""
            )
            if entry.skill is not None:
                print(f"${entry.name}  {entry.skill.description}  [{source}{override}]")
                continue
            invalid = _ui(self.language, "invalid", "无效")
            detail = entry.error.localized(self.language) if entry.error is not None else invalid
            print(f"${entry.name}  [{invalid}: {detail}]  [{source}{override}]")

    def print_mode(self, *, team_mode: bool, mcp_enabled: bool, tool_count: int) -> None:
        agent_shape = (
            _ui(self.language, "multi-agent", "多智能体")
            if team_mode
            else _ui(self.language, "single-agent", "单智能体")
        )
        tool_source = "MCP" if mcp_enabled else _ui(self.language, "built-in", "内置")
        message = _ui(
            self.language,
            f"[mode] {tool_source} {agent_shape} mode enabled with {tool_count} tool(s).",
            f"[模式] 已启用 {tool_source} {agent_shape} 模式，共 {tool_count} 个工具。",
        )
        print(message)

    def print_mcp_status(self, statuses: list[Any]) -> None:
        print(_ui(self.language, "[mcp] server status", "[mcp] 服务器状态"))
        for status in statuses:
            initialized = (
                _ui(self.language, "initialized", "已初始化")
                if status.initialized
                else _ui(self.language, "not initialized", "未初始化")
            )
            active = (
                _ui(self.language, "active", "活动")
                if status.active
                else _ui(self.language, "inactive", "非活动")
            )
            suffix = f" ({status.error})" if status.error else ""
            print(f"[mcp] {status.name}: {initialized}, {active}{suffix}")

    def print_mcp_enable_error(self, error: Exception, config_path: Any) -> None:
        print(
            _ui(
                self.language,
                f"mikucli: could not enable MCP mode: {error}",
                f"mikucli：无法启用 MCP 模式：{error}",
            ),
            file=sys.stderr,
        )
        print(
            _ui(
                self.language,
                f"mikucli: create {config_path} and try /mcp again.",
                f"mikucli：创建 {config_path} 后再尝试 /mcp。",
            )
        )

    def search_usage(self) -> str:
        return _ui(
            self.language,
            "mikucli: usage: /search <natural language query>",
            "mikucli：用法：/search <自然语言查询>",
        )

    def _progress_message(self, message: str) -> str:
        if self.language == "eng":
            return message
        if message == "Thinking....":
            return "思考中...."
        if message == "phase 1: planning":
            return "阶段 1：规划"
        if message == "phase 2: executing":
            return "阶段 2：执行"
        if message == "plan:":
            return "计划："
        compressed = re.fullmatch(
            r"Compressed (\d+) old session memory entries across (\d+) chunk\(s\); saved (\d+) long-term fact\(s\)\.",
            message,
        )
        if compressed:
            return f"已压缩 {compressed.group(1)} 条旧会话记忆，覆盖 {compressed.group(2)} 个片段；已保存 {compressed.group(3)} 条长期事实。"
        worker = re.fullmatch(r"(.+) executing \[(.+)\]: (.+)", message)
        if worker:
            return f"{worker.group(1)} 正在执行 [{worker.group(2)}]：{worker.group(3)}"
        reviewer = re.fullmatch(r"reviewer reviewing the results of \[(.+)\]", message)
        if reviewer:
            return f"审核智能体正在审核 [{reviewer.group(1)}] 的结果"
        approved = re.fullmatch(r"\[(.+)\] review approved", message)
        if approved:
            return f"[{approved.group(1)}] 审核通过"
        rejected = re.fullmatch(r"\[(.+)\] review rejected: (.+)", message)
        if rejected:
            return f"[{rejected.group(1)}] 审核未通过：{rejected.group(2)}"
        return message


def _truncate(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... truncated ..."
