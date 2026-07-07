# Changelog

## Unreleased

### Added

- Added eval suite result metrics for success rate, tool-call count, model retries, step retries, structured failure reasons, latency, provider-reported token cost, eval price, and estimated spend.
- Added machine-readable benchmark run summaries to JSON result files and human-readable Markdown reports under `.mikucli/evaluation/bench/runs/`.
- Added benchmark CLI flags for spend estimation: `--prompt-token-price-per-million` and `--completion-token-price-per-million`.
- Added `/eval run` as an interactive slash command that starts the eval suite benchmark harness with the active workspace and model settings.
- Documented eval suite terminology in `CONTEXT.md` and eval suite usage in `README.md`.

### Changed

- `BenchmarkRunner.run()` and `run_benchmarks()` now return `(results, json_path, markdown_report_path)`.
