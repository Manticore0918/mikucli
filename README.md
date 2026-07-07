# mikucli

mikucli is a local Python command-line agent runner. It runs an interactive Orchestrator-SubAgent session in a workspace, lets the agent use a small built-in tool set, applies static tool risk policy, asks for approval before medium- and high-risk tools run, and records session logs under `.mikucli/runs/`.

## Install for local development

```powershell
python -m pip install -e .
```

## Configure

By default, mikucli reads shared credentials and defaults from `~/.mikucli/.env`. If that file is missing
and no API key is available from higher-priority config, mikucli creates it from the default template and asks
you to fill in `BIGMODEL_API_KEY`, then restart.

You can also create it yourself:

```powershell
New-Item -ItemType Directory -Force ~/.mikucli
Copy-Item .env.example ~/.mikucli/.env
```

```dotenv
BIGMODEL_API_KEY=...
MIKUCLI_MODEL=glm-5.2
BIGMODEL_BASE_URL=https://open.bigmodel.cn/api/paas/v4/chat/completions
MIKUCLI_CONTEXT_WINDOW_TOKENS=128000
MIKUCLI_EMBEDDING_PROVIDER=ollama
MIKUCLI_EMBEDDING_MODEL=nomic-embed-text
MIKUCLI_OLLAMA_BASE_URL=http://localhost:11434
```

Only `BIGMODEL_API_KEY` is required. `MIKUCLI_MODEL`, `BIGMODEL_BASE_URL`, `MIKUCLI_CONTEXT_WINDOW_TOKENS`, `MIKUCLI_EMBEDDING_PROVIDER`, `MIKUCLI_EMBEDDING_MODEL`, and `MIKUCLI_OLLAMA_BASE_URL` are optional.

Workspace `.env` files are optional and merge over `~/.mikucli/.env` by key, so a workspace can override only
the settings it needs while reusing the global API key. You can also use environment variables:

```powershell
$env:BIGMODEL_API_KEY = "..."
$env:MIKUCLI_MODEL = "glm-5.2"
```

Config precedence is:

1. direct CLI flags like `--model` and `--context-window-tokens`
2. direct environment variables like `BIGMODEL_API_KEY` and `MIKUCLI_MODEL`
3. file named by `--env-file`
4. file named by `MIKUCLI_ENV_FILE`
5. workspace `.env`
6. user config file `~/.mikucli/.env`
7. built-in defaults

File-backed config merges by key from lower to higher priority. To use a different `.env` file:

```powershell
mikucli --env-file C:\Users\you\mikucli.env
```

## Run

```powershell
mikucli "inspect this project and suggest the next step"
mikucli --workspace D:\Personal_Projects\mikucli
mikucli --model glm-5.2
mikucli --context-window-tokens 128000
```

If no task prompt is provided, `mikucli` starts an interactive session and asks for the first prompt.

Interactive sessions start in built-in single-agent mode. Type `/team` to toggle multi-agent mode. Type `/mcp`
to toggle MCP mode. Type `/eval run` to run the eval suite benchmark harness in the foreground with per-case
completion details, `/eval run-back` to run it in the background, and `/eval stop` to stop a background run after
the current benchmark case and write a report. Type `/lang-chn` to show the terminal interface in Simplified
Chinese, and `/lang-eng` to switch it back to English. The mode toggles are independent, so the session can be in
built-in single-agent, built-in multi-agent, MCP single-agent, or MCP multi-agent mode.

When `/mcp` turns MCP mode on, mikucli starts the servers configured in `.mikucli/mcp.json`, validates the
configured tool bindings against each server's `tools/list` response, prints server status, and starts a fresh
session that exposes MCP tools instead of built-in tools. When `/mcp` turns MCP mode off, mikucli closes MCP
connections and starts a fresh session with built-in tools. `/team` toggles between single-agent and
multi-agent while preserving the current tool source.

MCP server status uses two terms:

- `initialized`: the MCP server completed initialization successfully
- `active`: the MCP server is currently responsive when status is shown

If `.mikucli/mcp.json` is missing or invalid, or a server fails to initialize, mikucli stays in built-in mode.
Every `/mcp` or `/team` toggle starts fresh active session memory; long-term memory remains shared.
Language switching affects only the current terminal session's interface text; it does not translate model answers,
tool output, saved memory, or run logs.

Use `/index` to build or refresh the local Codebase Index. Codebase Retrieval uses Ollama embeddings by default, so start Ollama and pull the embedding model first:

```powershell
ollama pull nomic-embed-text
```

Use `/search <natural language query>` to search the Codebase Index directly.

## Built-in tools

- `list_files`: low risk; list files inside the workspace and run automatically
- `read_file`: low risk; read a file inside the workspace and run automatically
- `write_file`: medium risk; show the proposed diff and ask for approval before writing
- `run_shell`: high risk; ask for approval before executing a shell command
- `save_long_term_memory`: low risk; save a deduplicated memory for future sessions in the workspace and run automatically
- `search_codebase`: low risk; search the local Codebase Index for relevant source and documentation chunks and run automatically

## Eval suite

Run the eval suite benchmark harness with:

```powershell
python -m mikucli.evaluation.bench --workspace D:\Personal_Projects\mikucli
```

Inside an interactive `mikucli` session, type `/eval run` to run the same benchmark harness with the active
workspace, model, and context-window settings. Foreground eval prints one line when each benchmark case finishes,
using `MISSION SUCCEED` or `MISSION FAILED` with retry/tool metrics. Type `/eval run-back` to run the eval suite in the background.
Type `/eval stop` to request a cooperative stop; mikucli finishes the current benchmark case, then writes JSON and
Markdown reports for the completed cases.

Use `--list` to list benchmark cases and `--case <case-id>` to run selected cases. Each run writes machine-readable
JSON and a human-readable Markdown report under `.mikucli/evaluation/bench/runs/`. Reports include success rate,
tool-call count, model retries, step retries, structured failure reasons, total latency, agent latency, LLM latency,
and eval cost as provider-reported prompt, completion, and total tokens.

To estimate spend, pass eval prices as money cost per one million tokens:

```powershell
python -m mikucli.evaluation.bench `
  --prompt-token-price-per-million 2 `
  --completion-token-price-per-million 4
```

When prices are omitted, reports still record token cost and leave estimated spend unknown.

## MCP mode

Configure MCP mode with `.mikucli/mcp.json`:

```json
{
  "servers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]
    },
    "zread": {
      "command": "zread-mcp",
      "args": [],
      "env": {
        "ZREAD_TOKEN": "..."
      }
    }
  },
  "tools": {
    "read_workspace_file": {
      "server": "filesystem",
      "mcp_tool_name": "read_file",
      "risk": "low",
      "read_only": true
    },
    "read_github_file": {
      "server": "zread",
      "mcp_tool_name": "read_file",
      "risk": "low",
      "read_only": true
    }
  }
}
```

Each entry under `tools` is a model-facing tool name. The binding routes to one server and one MCP tool through
`server` and `mcp_tool_name`. `risk` is optional and defaults to `high`; supported values are `low`, `medium`,
and `high`. Low-risk MCP calls run automatically. Medium- and high-risk MCP calls use the same terminal approval
flow as built-in tools.

`read_only` is optional and defaults to `false`. In MCP multi-agent mode, worker subagents receive all configured
MCP tools. Planner and reviewer subagents receive only MCP tool bindings marked with `read_only: true`.

MCP servers are started with the workspace as their working directory. The first implementation supports
`command`, `args`, and optional `env` for server launch configuration.

## Multi-agent roster

mikucli starts the orchestrator as the main agent. By default it initializes four subagents:

- `planner-1`: breaks down tasks, identifies dependencies, and proposes execution plans
- `worker-1`: executes implementation or investigation work
- `worker-2`: executes implementation or investigation work
- `reviewer-1`: checks work for defects, missed requirements, and verification gaps

Only the orchestrator can delegate to subagents. Worker subagents use the constrained workspace tools and tool approval flow; planner and reviewer subagents receive only `list_files` and `read_file`.

## Orchestrator workflow

Every user turn follows the same workflow:

1. `planner-1` receives the task and returns a JSON execution plan.
2. The orchestrator translates that plan into `ExecutionStep` objects and builds dependency relations.
3. Dependency-ready steps are assigned to workers. Independent steps in the same dependency batch can run simultaneously, bounded by the two initialized workers. When a step depends on completed steps, the worker receives dependency context capped at the first 500 characters.
4. `reviewer-1` reviews each completed step description and worker result. The reviewer returns JSON with `approved`, `summary`, `issues`, and `suggestions`; rejected review issues and suggestions are returned to the worker for another attempt.
5. Worker and reviewer chat histories are cleared after each completed step so later steps start with clean subagent context.
6. Steps blocked by failed or skipped dependencies are marked skipped and reported to the user.
7. The orchestrator writes the step statuses and summarized results into session memory, then returns the execution summary.

## Notes

- Active session memory lives only in the current process.
- Session memory keeps recent entries in FIFO order, retains old entries for compression, and starts compression when token usage exceeds 80% of the configured context window.
- Memory retrieval ranks session and long-term memories before each model request using keyword overlap, linear 24-hour time decay from 1.0 to 0.5, and a 1.2x source multiplier for long-term memories.
- Context compression keeps the latest 3 user chat rounds verbatim, summarizes older memory with an LLM map-reduce pass, and extracts durable facts into long-term memory.
- Long-term memory persists in `.mikucli/long_term_memory.json`, is loaded on startup, deduplicates saved facts, and keeps the original timestamp when duplicate content is saved again.
- Run logs persist under `.mikucli/runs/`.
- The BigModel client prefers native tool calling. If the provider response does not include tool calls, the runner accepts a strict JSON action fallback.
- Tool risk levels are stored in a runner-side `ToolPolicy`; they are enforced by the tool registry and are not exposed to the model in tool schemas.
- Codebase Retrieval stores its SQLite index under `.mikucli/codebase_index/`.
- `/index` performs a full rebuild in v1, writing a temporary database first and replacing the active index only after successful embedding and validation.
- Code chunks for Python and Java use tree-sitter. Markdown, XML, TOML, and other non-code text files use language-neutral 2000-character chunks.
- Hybrid search combines Ollama embedding cosine similarity with SQLite FTS/BM25 using reciprocal rank fusion.
