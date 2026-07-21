from __future__ import annotations

from mikucli.config import Config
from mikucli.console import TerminalConsole
from mikucli.llm import BigModelClient
from mikucli.mcp_config import load_mcp_config
from mikucli.mcp_tools import McpToolSet
from mikucli.memory import LongTermMemory
from mikucli.multi_agent import OrchestratorSession
from mikucli.react import AgentSession


def new_single_agent_session(
    *,
    client: BigModelClient,
    config: Config,
    tools: object,
    console: TerminalConsole,
    max_steps: int,
    long_term_memory: LongTermMemory,
) -> AgentSession:
    return AgentSession(
        client=client,
        model=config.model,
        workspace=config.workspace,
        tools=tools,
        console=console,
        max_steps=max_steps,
        context_window_tokens=config.context_window_tokens,
        long_term_memory=long_term_memory,
    )


def new_session(
    *,
    client: BigModelClient,
    config: Config,
    tools: object,
    console: TerminalConsole,
    max_steps: int,
    long_term_memory: LongTermMemory,
    team_mode: bool,
) -> AgentSession | OrchestratorSession:
    if team_mode:
        return OrchestratorSession(
            client=client,
            model=config.model,
            workspace=config.workspace,
            tools=tools,
            console=console,
            max_steps=max_steps,
            context_window_tokens=config.context_window_tokens,
            long_term_memory=long_term_memory,
        )
    return new_single_agent_session(
        client=client,
        config=config,
        tools=tools,
        console=console,
        max_steps=max_steps,
        long_term_memory=long_term_memory,
    )


def connect_mcp_tools(*, config: Config, console: TerminalConsole) -> McpToolSet:
    mcp_tools = McpToolSet.connect(
        config=load_mcp_config(config.workspace),
        workspace=config.workspace,
        confirm_tool=console.confirm_tool,
    )
    console.print_mcp_status(mcp_tools.statuses())
    return mcp_tools
