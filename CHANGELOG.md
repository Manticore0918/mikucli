# Changelog

## Unreleased

### Added

- Added 21 credential-free workflow contracts with explicit expected outcomes, cancellation isolation, persistence/MCP recovery probes, and both-runtime coverage.
- Added seven tasks from a checksummed, pinned repository snapshot, held-out behavior checks with parent-side scoring, and negative scorer validation.
- Added per-case reliability evidence, strict versioned release policies, a failing `gate` command, and bounded live `campaign` profiles with compatible-baseline preflight.
- Added Ubuntu/Windows offline eval gates and opt-in scheduled/manual live-provider workflows that upload only allowlisted evidence summaries.

- Added explicit task outcomes, meaningful CLI exit codes, one-shot `--json` results, and `--team` startup. Run logs, traces, and benchmark checks now retain unsuccessful runtime outcomes.
- Added cancellable provider/embedding retries, bounded request deadlines and response bodies, dependency preflight through `--check`, and shared per-task duration, reported-token, estimated-spend, and tool-output limits.
- Added transactional SQLite long-term memory with process-safe deduplication, one-time legacy JSON migration, verified automatic snapshots, and credential-free offline backup/restore commands through `python -m mikucli.memory`.
- Added durable-memory regression tests for concurrent processes and worker threads, killed writes and migrations, failed commits, corruption, bounded lock waits, and interrupted recovery.
- Added explicit `$skill-name` task invocation with user and version-controlled workspace `SKILL.md` scopes, per-turn system overlays, multi-agent propagation, `/skills` discovery, validation, and run/trace metadata.
- Added optional `start_line` and `end_line` arguments to `read_file`, with bounded large-file handling that directs agents through Codebase Retrieval before exact ranged reads.
- Added eval suite result metrics for success rate, tool-call count, model retries, step retries, structured failure reasons, latency, provider-reported token cost, eval price, and estimated spend.
- Added machine-readable benchmark run summaries to JSON result files and human-readable Markdown reports under `.mikucli/evaluation/bench/runs/`.
- Added benchmark CLI flags for spend estimation: `--prompt-token-price-per-million` and `--completion-token-price-per-million`.
- Added `/eval run` as an interactive slash command that starts the eval suite benchmark harness with the active workspace and model settings.
- Added `/eval run-single` and `/eval run-multi` to run the six benchmark cases for one agent mode at a time.
- Added foreground eval mission reporting: `/eval run` now prints each completed benchmark case as `MISSION SUCCEED` or `MISSION FAILED` with metrics.
- Added `/eval run-back` as the background eval suite command.
- Added `/eval stop` as an interactive slash command that requests a cooperative eval suite stop and writes reports for completed benchmark cases.
- Reported eval runtime as a single end-to-end latency metric in JSON and Markdown reports.
- Added repeated benchmark trials with `--repeat`, deterministic trial ordering with `--shuffle-seed`, isolated per-trial workspaces, and trial-aware observability records.
- Added median and p95 latency statistics plus Wilson 95% confidence intervals for benchmark pass rates.
- Added versioned benchmark reports with suite, configuration, environment, and sandbox-capability metadata; run comparisons now flag incompatible or missing fingerprints.
- Added atomic per-trial checkpoints with explicit `running`, `stopped`, `interrupted`, and `complete` states, plus compatible-run recovery through `--resume`.
- Added `--require-os-sandbox` so callers can fail closed when the available benchmark isolation backend is not enforced by the operating system.
- Added strict versioned benchmark report validation for resume and observability import, including schema, run identity, trial uniqueness, numeric bounds, and summary/result consistency checks.
- Added benchmark catalog validation for stable unique task IDs, callable setup/check functions, and non-duplicated session-mode coverage.
- Added a subprocess smoke test that boots the mikucli CLI with a temporary env file.
- Documented eval suite terminology in `CONTEXT.md` and eval suite usage in `README.md`.

### Changed

- Generalized model configuration around `MIKUCLI_API_KEY`, `MIKUCLI_MODEL`, and `MIKUCLI_BASE_URL`, added direct CLI overrides, and retained `BIGMODEL_*` as backward-compatible aliases.
- Applied the required Nomic `search_document:` and `search_query:` task prefixes to semantic retrieval, while preserving raw queries for lexical search and rejecting indexes built with incompatible embedding profiles.
- Serialized non-read-only and approval-requiring tool calls across multi-agent workers while preserving concurrent approval-free read-only inspection.
- `BenchmarkRunner.run()` and `run_benchmarks()` now return `(results, json_path, markdown_report_path)`.
- Benchmark summaries now distinguish planned, completed, and incomplete trials instead of presenting partial runs as complete suites.
- Repeated-run comparisons now aggregate trials by case using pass rates and median metrics, with explicit `flaky_failure` classification.
- Benchmark check execution now uses bounded output capture, scrubbed environments, whole-process-tree timeouts, and an explicit capability contract that identifies the current Python audit-hook backend as non-OS-enforced.
- Split benchmark lifecycle, persistence, metadata, and isolated-process responsibilities into focused modules within the production source line budget.

### Fixed

- Fixed step exhaustion and failed team workflows reporting success, planner/worker failures being consumed as successful content, and LangGraph's default recursion limit ending valid longer tool loops.
- Fixed unbounded shell-output capture and cancellation gaps during retry, approval, serialized mutation, and MCP waits.
- Fixed independent agent sessions overwriting each other's durable facts and failed saves leaving unsaved facts cached as duplicates. Corrupt storage now warns without preventing ordinary CLI use; failed fact batches preserve compression sources for retry.
- Fixed benchmark function scoring so expected values and equality decisions remain in the parent process; submission-defined equality objects and `__main__` monkeypatches can no longer forge a pass.
- Fixed isolated unittest scoring so test modules cannot replace `TextTestRunner.run` during discovery and report a broken suite as successful.
- Fixed cooperative stops being persisted as completed trials; stopped in-flight trials now remain pending and are rerun from a clean fixture workspace on resume.
- Fixed shell-process cancellation returning before the Windows command process was reaped, which could leave `/eval stop` workers and workspace handles alive.
- Fixed required-tool checks accepting failed calls and failed-tool recovery being inferred solely from unrelated task success.
- Fixed model retry metrics counting failed tool calls themselves instead of subsequent model turns after a failed tool result.
- Fixed configuration fingerprints omitting provider endpoints, memory retrieval weights, embedding settings, and time-decay parameters while continuing to exclude credentials and URL query data.
- Fixed observability comparisons classifying regressions between incompatible runs; incompatible case rows are now explicitly marked and the dashboard suppresses deltas.
- Fixed stored run, case, trace, check, and tool content being interpolated into dashboard HTML; dynamic cells and controls now use DOM text nodes and encoded request paths.
- Fixed background eval lifecycle handling so cooperative stops and interrupted runs retain completed results and write an accurate terminal checkpoint.
- Fixed isolated function checks accepting spoofed result markers or payloads from abnormally terminated processes.
- Fixed isolated test discovery treating a run with zero discovered tests as successful.
- Fixed benchmark checks leaking unbounded subprocess output or leaving descendant processes alive after a timeout.
- Fixed workspace snapshots excluding `.mikucli` changes, which could hide unintended agent mutations.
- Fixed tool cleanup exceptions being omitted from case failures.
- Fixed structured JSON change checks mishandling list and mapping values.
- Fixed ordered path-scope negations being evaluated with incorrect precedence.
- Fixed negative and non-finite eval prices being accepted.
- Fixed repeated trials colliding in observability storage and benchmark comparisons.
- Fixed resume accepting duplicate, out-of-range, foreign-case, or count-inconsistent trial records.
- Fixed retry comparisons treating any increase as a regression instead of the documented zero-to-non-zero threshold.
