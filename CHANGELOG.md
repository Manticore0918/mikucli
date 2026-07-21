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
- Split eval latency reporting into total latency, agent latency, and LLM latency in JSON and Markdown reports.
- Added a subprocess smoke test that boots the mikucli CLI with a temporary env file.
- Documented eval suite terminology in `CONTEXT.md` and eval suite usage in `README.md`.

### Changed

- Applied the required Nomic `search_document:` and `search_query:` task prefixes to semantic retrieval, while preserving raw queries for lexical search and rejecting indexes built with incompatible embedding profiles.
- Serialized non-read-only and approval-requiring tool calls across multi-agent workers while preserving concurrent approval-free read-only inspection.
- `BenchmarkRunner.run()` and `run_benchmarks()` now return `(results, json_path, markdown_report_path)`.
