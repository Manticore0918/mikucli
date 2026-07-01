# mikucli

mikucli is a local Python command-line agent runner. It runs an interactive Orchestrator-SubAgent session in a workspace, lets the agent use a small built-in tool set, requires review before shell commands, writes files directly, prints concise diffs after file changes, and records session logs under `.mikucli/runs/`.

## Install for local development

```powershell
python -m pip install -e .
```

## Configure

Create `.env` in your workspace:

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

You can also use environment variables:

```powershell
$env:BIGMODEL_API_KEY = "..."
$env:MIKUCLI_MODEL = "glm-5.2"
```

Environment variables override the `.env` file. `--model` overrides both `MIKUCLI_MODEL` and the `.env` file model. To use a different `.env` file:

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

Interactive sessions start in single-agent mode. Type `/team` to switch the current CLI session to multi-agent mode.

Use `/index` to build or refresh the local Codebase Index. Codebase Retrieval uses Ollama embeddings by default, so start Ollama and pull the embedding model first:

```powershell
ollama pull nomic-embed-text
```

Use `/search <natural language query>` to search the Codebase Index directly.

## Built-in tools

- `list_files`: list files inside the workspace
- `read_file`: read a file inside the workspace
- `write_file`: write a file inside the workspace and show a concise diff after applying it
- `run_shell`: request a shell command; every command requires user review before execution
- `save_long_term_memory`: save a deduplicated memory for future sessions in the workspace
- `search_codebase`: search the local Codebase Index for relevant source and documentation chunks

## Multi-agent roster

mikucli starts the orchestrator as the main agent. By default it initializes four subagents:

- `planner-1`: breaks down tasks, identifies dependencies, and proposes execution plans
- `worker-1`: executes implementation or investigation work
- `worker-2`: executes implementation or investigation work
- `reviewer-1`: checks work for defects, missed requirements, and verification gaps

Only the orchestrator can delegate to subagents. Worker subagents use the constrained workspace tools and command review flow; planner and reviewer subagents receive only `list_files` and `read_file`.

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
- Codebase Retrieval stores its SQLite index under `.mikucli/codebase_index/`.
- `/index` performs a full rebuild in v1, writing a temporary database first and replacing the active index only after successful embedding and validation.
- Code chunks for Python and Java use tree-sitter. Markdown, XML, TOML, and other non-code text files use language-neutral 2000-character chunks.
- Hybrid search combines Ollama embedding cosine similarity with SQLite FTS/BM25 using reciprocal rank fusion.
