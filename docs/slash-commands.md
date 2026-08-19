# Slash Command Manual

Slash Commands control a running mikucli session without sending a Task Prompt to the agent. Start an interactive session by launching `mikucli` without a Task Prompt, then type a command in the Input Composer and press `Enter`.

Command names and arguments are case-sensitive. Leading and trailing whitespace is ignored.

## Quick reference

| Command | Description |
| --- | --- |
| `/dashboard` | Start or reuse the local observability dashboard and open it in the default browser. |
| `/eval run` | Run all twelve evaluation cases in the foreground. |
| `/eval run-single` | Run the six single-agent evaluation cases with the runtime selected at startup. |
| `/eval run-multi` | Run the six multi-agent evaluation cases in the foreground. |
| `/eval run-back` | Run all twelve evaluation cases in the background while continuing to use the agent. |
| `/eval stop` | Request a cooperative stop of the active evaluation suite. |
| `/exit` | Exit the interactive session. |
| `/quit` | Alias for `/exit`. |
| `/index` | Build or completely refresh the local Codebase Index. |
| `/lang-chn` | Switch the terminal interface to Simplified Chinese. |
| `/lang-eng` | Switch the terminal interface to English. |
| `/mcp` | Toggle between built-in tools and configured MCP tools. |
| `/search <query>` | Search the local Codebase Index using a natural-language query. |
| `/skills` | List available Skills, their descriptions, and their sources. |
| `/stop` | Stop the current agent task or, when the agent is idle, another active process. |
| `/team` | Toggle between single-agent and multi-agent mode. |
| `/tui fullscreen` | Switch the current session to the full-screen Terminal UI. |
| `/tui inline` | Switch the current session to the inline Terminal UI. |

## Session and process control

### `/stop`

Stop the first active target in this order:

1. The current agent task, model request, tool call, shell command, or pending Tool Approval.
2. A running evaluation suite.
3. The dashboard backend started by mikucli.

If nothing is running, mikucli reports that there is no process to stop. During an agent task, pressing `Ctrl+C` has the same Task Stop behavior. To target an evaluation suite while an agent task is also running, use `/eval stop`.

### `/exit` and `/quit`

Exit an idle interactive session. The two commands are equivalent. If a background evaluation is active, mikucli requests a stop, waits for the partial report, restores the terminal, and then exits.

In inline mode, `Ctrl+D` also exits. In the full-screen Terminal UI, press `Ctrl+D` from an empty Input Composer.

### `/tui fullscreen` and `/tui inline`

Switch the renderer for the current interactive session:

- `fullscreen` uses the scrollable Activity View, persistent Session Status, and bottom-anchored Input Composer. It requires interactive stdin and stdout.
- `inline` prints the session as a conventional terminal stream and is the fallback for redirected or unsupported terminals.

The choice is not saved for later launches. Typing `/tui` without a valid argument prints the accepted forms.

## Agent and tool modes

### `/team`

Toggle between single-agent mode and multi-agent mode. Multi-agent mode uses an orchestrator to coordinate planner, worker, and reviewer roles.

The current tool source is preserved: `/team` works with either built-in tools or MCP tools. Each toggle starts fresh active session memory; long-term memory remains shared.

### `/mcp`

Toggle between built-in tools and the MCP servers configured in `.mikucli/mcp.json`.

When enabling MCP mode, mikucli starts the configured servers, validates their advertised tools, and creates a fresh agent session exposing MCP tools. If the configuration is missing or invalid, or a server cannot initialize, mikucli stays in built-in mode. Toggling MCP mode off closes the MCP connections and creates a fresh session with built-in tools.

Each toggle starts fresh active session memory; long-term memory remains shared. The current single-agent or multi-agent selection is preserved.

## Codebase Retrieval

### `/index`

Build or completely refresh the Codebase Index stored under `.mikucli/codebase_index/`. The rebuild is prepared in a temporary database and replaces the active index only after it succeeds.

Codebase Retrieval uses the configured embedding provider. With the default configuration, Ollama must be running and the embedding model must be available:

```powershell
ollama pull nomic-embed-text
```

### `/search <query>`

Search the Codebase Index directly with a natural-language query and print the best matching chunks.

```text
/search how are tool approvals enforced?
```

Run `/index` before the first search and whenever mikucli reports that the index format or embedding profile has changed. Typing `/search` without a query prints the command usage.

## Evaluation and observability

### `/eval run`

Run all twelve benchmark cases in the foreground: six single-agent cases and six multi-agent cases. mikucli prints case start and completion details followed by the result and report paths. In an interactive session, the run remains stoppable but is exclusive: ordinary Task Prompts and unrelated Slash Commands are rejected until it finishes.

### `/eval run-single`

Run only the six single-agent benchmark cases in the foreground. The selection includes each task's built-in or MCP
single-agent case as appropriate. When mikucli starts with `--runtime langgraph`, each case receives a fresh isolated
`GraphAgentSession`; with `--runtime native`, each case receives a native `AgentSession`.

### `/eval run-multi`

Run only the six multi-agent benchmark cases in the foreground. The selection includes each task's built-in or MCP multi-agent case as appropriate.

### `/eval run-back`

Run all twelve benchmark cases in the background. Agent work can continue while the suite is active, and completion, failure, and report paths are printed automatically.

This form is available only in an interactive session.

### `/eval stop`

Request a cooperative stop of the evaluation suite. The active case stops at its next safe cancellation point, after which mikucli writes partial JSON and Markdown reports and prints the completion summary.

Typing `/eval` without a valid subcommand prints the accepted forms.

### `/dashboard`

Start the local observability dashboard backend at `http://127.0.0.1:8765/` and open it in the default browser. If it is already running, mikucli reuses it. When no agent task or evaluation suite is active, `/stop` stops the dashboard backend.

## Language and Skills

### `/lang-chn` and `/lang-eng`

Switch the terminal interface between Simplified Chinese and English for the current session. This does not translate model answers, tool output, saved memory, or Run Logs.

### `/skills`

List all valid User and Workspace Skills, including each Skill's description and source. Workspace Skills override same-named User Skills.

Skills are invoked with a dollar-sign prefix, not a Slash Command:

```text
$skill-name describe the task
```

## Availability while work is running

While a foreground agent task or foreground evaluation is active, the Input Composer accepts only:

- `/stop`
- `/eval stop`
- `y`, `yes`, `n`, or `no` when a Tool Approval is pending

Other Task Prompts and Slash Commands become available again when the foreground work is idle. A background evaluation started with `/eval run-back` does not block agent work.

Commands are intended primarily for interactive sessions. `/index`, `/search <query>`, `/skills`, `/dashboard`, `/eval run`, `/eval run-single`, and `/eval run-multi` may also be supplied as the sole command-line Task Prompt, for example `mikucli "/index"`. Session controls such as `/tui`, `/exit`, and `/quit` are interactive-only.
