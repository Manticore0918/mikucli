# Changelog

## Unreleased

### Added

- Added explicit `$skill-name` task invocation with user and version-controlled workspace `SKILL.md` scopes, per-turn system overlays, multi-agent propagation, `/skills` discovery, validation, and run/trace metadata.
- Added optional `start_line` and `end_line` arguments to `read_file`, with bounded large-file handling that directs agents through Codebase Retrieval before exact ranged reads.
- Added eval suite result metrics for success rate, tool-call count, model retries, step retries, structured failure reasons, latency, provider-reported token cost, eval price, and estimated spend.
- Added machine-readable benchmark run summaries to JSON result files and human-readable Markdown reports under `.mikucli/evaluation/bench/runs/`.
- Added benchmark CLI flags for spend estimation: `--prompt-token-price-per-million` and `--completion-token-price-per-million`.
- Added `/eval run` as an interactive slash command that starts the eval suite benchmark harness with the active workspace and model settings.
- Added foreground eval mission reporting: `/eval run` now prints each completed benchmark case as `MISSION SUCCEED` or `MISSION FAILED` with metrics.
- Added `/eval run-back` as the background eval suite command.
- Added `/eval stop` as an interactive slash command that requests a cooperative eval suite stop and writes reports for completed benchmark cases.
- Reported eval runtime as a single end-to-end latency metric in JSON and Markdown reports.
- Added repeated benchmark trials with `--repeat`, deterministic trial ordering with `--shuffle-seed`, isolated per-trial workspaces, and trial-aware observability records.
- Added median and p95 latency statistics plus Wilson 95% confidence intervals for benchmark pass rates.
- Added versioned benchmark reports with suite, configuration, environment, and sandbox-capability metadata; run comparisons now flag incompatible or missing fingerprints.
- Added atomic per-trial checkpoints with explicit `running`, `stopped`, `interrupted`, and `complete` states, plus compatible-run recovery through `--resume`.
- Added `--require-os-sandbox` so callers can fail closed when the available benchmark isolation backend is not enforced by the operating system.
- Added a subprocess smoke test that boots the mikucli CLI with a temporary env file.
- Documented eval suite terminology in `CONTEXT.md` and eval suite usage in `README.md`.

### Changed

- Applied the required Nomic `search_document:` and `search_query:` task prefixes to semantic retrieval, while preserving raw queries for lexical search and rejecting indexes built with incompatible embedding profiles.
- Serialized non-read-only and approval-requiring tool calls across multi-agent workers while preserving concurrent approval-free read-only inspection.
- `BenchmarkRunner.run()` and `run_benchmarks()` now return `(results, json_path, markdown_report_path)`.
- Benchmark summaries now distinguish planned, completed, and incomplete trials instead of presenting partial runs as complete suites.
- Benchmark check execution now uses bounded output capture, scrubbed environments, whole-process-tree timeouts, and an explicit capability contract that identifies the current Python audit-hook backend as non-OS-enforced.
- Split benchmark lifecycle, persistence, metadata, and isolated-process responsibilities into focused modules within the production source line budget.

### Fixed

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
