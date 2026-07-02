# mikucli

mikucli is a local command-line agent context for running task-oriented AI assistance against a user's workspace.

## Language

**Agent Runner**:
A command-line assistant that accepts a task, coordinates model reasoning with constrained local tools, and reports progress and outcomes to the user.
_Avoid_: Chat wrapper, bot

**CLI Command**:
The installed terminal command users run to start an agent session, named `mikucli`.
_Avoid_: Binary, app name

**Slash Command**:
A user-facing interactive session control that starts with `/` and is handled by the agent runner instead of being sent as a task prompt.
_Avoid_: CLI command, tool request, shell command

**Task Prompt**:
The user's initial instruction for an agent session, including the desired outcome and any constraints the user provides up front.
_Avoid_: Chat message, query

**Agent Session**:
An interactive chat between the user and the agent runner where the agent can reason, request tools, report progress, and continue across multiple user turns.
_Avoid_: Single-shot run, conversation log

**Orchestrator**:
The main agent in an agent session that coordinates the user-facing work and remains responsible for the final answer.
_Avoid_: Manager agent, controller, lead bot

**SubAgent**:
A role-specific agent that receives focused delegated tasks from the orchestrator and returns a concise result.
_Avoid_: Helper bot, child process, plugin

**Planner SubAgent**:
A subagent responsible for breaking down tasks, clarifying dependencies, and proposing execution plans.
_Avoid_: Strategist, architect bot

**Execution Plan**:
A JSON plan from the planner subagent that describes the ordered work needed to satisfy a task prompt.
_Avoid_: Todo list, checklist, outline

**ExecutionStep**:
One unit of work translated from an execution plan, with its dependency relationships and completion status tracked by the orchestrator.
_Avoid_: Job, task item, work unit

**Worker SubAgent**:
A subagent responsible for implementation, investigation, and other concrete workspace work delegated by the orchestrator.
_Avoid_: Executor, coder bot

**Reviewer SubAgent**:
A subagent responsible for checking completed or proposed work for defects, missed requirements, and verification gaps.
_Avoid_: Critic bot, QA bot

**Workspace**:
The explicit directory where an agent run is allowed to inspect files, change files, and execute commands unless the user grants a broader path.
_Avoid_: Project folder, sandbox, repo

**Tool**:
A named built-in capability the agent can request through a strict interface, starting with file reading, file writing, file listing, and shell execution.
_Avoid_: Plugin, arbitrary code hook, extension

**MCP Mode**:
An agent session mode where the agent runner exposes configured MCP tools instead of built-in tools.
_Avoid_: Plugin mode, external mode

**MCP Server**:
A configured external tool provider that the agent runner can connect to during MCP mode.
_Avoid_: Plugin, subagent

**MCP Tool Binding**:
A unique internal mapping from a model-facing tool name to one tool exposed by one MCP server, including the tool's risk classification.
_Avoid_: Tool rename, alias only

**Model-Facing Tool Name**:
The clear tool name shown to the model for an MCP tool binding.
_Avoid_: Internal id, server tool name

**Tool Request**:
A validated request from the agent to invoke one of the available tools, represented through native provider tool calling when available or a strict JSON action format otherwise.
_Avoid_: Function call, command, raw model output

**Tool Risk Level**:
A static classification that determines how much human approval a tool request needs before it runs: low risk runs automatically, medium risk requests approval before mutation, and high risk requires approval before execution.
_Avoid_: Permission level, safety score

**ToolPolicy**:
The manually defined policy layer that stores each built-in tool's risk level for the agent runner.
_Avoid_: Dynamic risk scoring, model-facing safety metadata

**Tool Approval**:
The user-facing decision point before a medium-risk or high-risk tool request runs.
_Avoid_: Command review, permission prompt, security check

**LLM Provider**:
The external model service an agent runner uses to reason about a task and decide which tools to request.
_Avoid_: Backend, AI engine, vendor

**Model**:
The named LLM chosen for an agent run, supplied by the configured LLM provider and selectable by the user.
_Avoid_: Engine, preset

**Change Summary**:
The concise diff shown after the agent has applied file changes, so the user can inspect what changed during the run.
_Avoid_: File review, approval diff

**Run Log**:
The local record of an agent session, including the task prompt, model, workspace, tool activity, tool approval outcomes, changed paths, and final answer.
_Avoid_: Transcript, history, audit database

**Session Memory**:
The active conversational state used during one running agent session, discarded when the process exits.
_Avoid_: Short-Term Memory, persistent memory, resume state

**Long-Term Memory**:
A persistent set of deduplicated facts the agent runner can carry across agent sessions in the same workspace.
_Avoid_: Session Memory, run log, transcript

**Codebase Retrieval**:
The agent runner capability that finds relevant workspace source or documentation context for a task prompt.
_Avoid_: RAG, memory, run log lookup

**Codebase Index**:
The persisted, searchable representation of workspace source and documentation used by codebase retrieval.
_Avoid_: Vector memory, knowledge base, transcript index

**Code Chunk**:
A searchable excerpt of a workspace file stored in the codebase index with enough location metadata to trace it back to source.
_Avoid_: Memory entry, document, snippet

**Hybrid Search**:
Codebase retrieval that combines lexical matching with semantic similarity instead of relying on only one ranking signal.
_Avoid_: Vector search, grep, keyword search

**ReAct Loop**:
The agent session pattern where the model alternates between deciding the next step, requesting a tool when needed, observing the result, and continuing until it can respond to the user.
_Avoid_: Free-form chat, autonomous script

**Progress Message**:
A concise user-facing update that describes visible agent activity without exposing raw model reasoning.
_Avoid_: Chain-of-thought, internal reasoning, debug trace
