from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from typing import Any

from .compare import compare_runs
from .store import LocalTraceStore


def response_for(path: str, query: dict[str, list[str]], store: LocalTraceStore) -> tuple[int, str, bytes]:
    if path == "/":
        return 200, "text/html; charset=utf-8", _dashboard_html().encode("utf-8")
    if path == "/runs":
        return _json_response({"runs": _runs(store)})
    if path.startswith("/runs/"):
        parts = path.strip("/").split("/")
        if len(parts) == 2:
            run = store.fetch_one("select * from runs where run_id = ?", (parts[1],))
            return _json_response({"run": _decode_row(run)})
        if len(parts) == 3 and parts[2] == "cases":
            return _json_response({"cases": _cases(store, parts[1])})
    if path.startswith("/cases/"):
        case_id = path.removeprefix("/cases/")
        case = store.fetch_one("select * from eval_cases where case_result_id = ?", (case_id,))
        if case is None:
            return _json_response({"error": "case not found"}, status=404)
        checks = store.fetch_all("select * from eval_checks where case_result_id = ? order by id", (case_id,))
        tools = store.fetch_all("select * from tool_calls where case_result_id = ? order by id", (case_id,))
        approvals = store.fetch_all("select * from approvals where case_result_id = ? order by id", (case_id,))
        return _json_response({"case": _decode_row(case), "checks": _decode_rows(checks), "tool_calls": _decode_rows(tools), "approvals": _decode_rows(approvals)})
    if path.startswith("/traces/"):
        parts = path.strip("/").split("/")
        if len(parts) == 2:
            trace = store.fetch_one("select * from traces where trace_id = ?", (parts[1],))
            return _json_response({"trace": _decode_row(trace)})
        if len(parts) == 3 and parts[2] == "spans":
            spans = store.fetch_all("select * from spans where trace_id = ? order by started_at", (parts[1],))
            events = store.fetch_all("select * from span_events where trace_id = ? order by at", (parts[1],))
            return _json_response({"spans": _decode_rows(spans), "events": _decode_rows(events)})
    if path == "/compare":
        base = _first(query, "base")
        head = _first(query, "head")
        if not base or not head:
            return _json_response({"error": "base and head query parameters are required"}, status=400)
        return _json_response(compare_runs(store, base, head))
    if path == "/failures":
        limit = _int_query(query, "limit", 100)
        rows = store.fetch_all("select * from eval_cases where passed = 0 order by case_result_id desc limit ?", (limit,))
        return _json_response({"failures": _decode_rows(rows)})
    return _json_response({"error": "not found"}, status=404)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mikucli.observability.api")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--store-root", default=str(Path.cwd() / ".mikucli" / "observability"))
    parser.add_argument("--no-auto-import", action="store_true", help="Do not auto-import benchmark JSON reports on startup.")
    args = parser.parse_args(argv)
    store_root = Path(args.store_root)
    store = LocalTraceStore(store_root, mode="sqlite")
    if not args.no_auto_import:
        imported = auto_import_benchmark_reports(store_root, store)
        if imported:
            print(f"mikucli observability: imported {imported} benchmark report(s)")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            status, content_type, body = response_for(parsed.path, parse_qs(parsed.query), store)
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            return

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"mikucli observability dashboard: http://{args.host}:{args.port}/")
    server.serve_forever()
    return 0


def _runs(store: LocalTraceStore) -> list[dict[str, Any]]:
    rows = store.fetch_all("select * from runs order by started_at desc")
    runs = _decode_rows(rows)
    for run in runs:
        if run is None:
            continue
        run_id = str(run.get("run_id") or "")
        run["traced_cases"] = store.fetch_all(
            "select count(*) as n from eval_cases where run_id = ? and trace_id != ''",
            (run_id,),
        )[0]["n"]
    return runs


def _cases(store: LocalTraceStore, run_id: str) -> list[dict[str, Any]]:
    rows = store.fetch_all("select * from eval_cases where run_id = ? order by case_id", (run_id,))
    return _decode_rows(rows)


def _decode_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_decode_row(row) for row in rows]


def _decode_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    decoded = dict(row)
    for key in list(decoded):
        if key.endswith("_json"):
            decoded[key.removesuffix("_json")] = _loads(str(decoded.pop(key)))
    for key in ("passed", "approved"):
        if key in decoded:
            decoded[key] = bool(decoded[key])
    return decoded


def _json_response(payload: Any, *, status: int = 200) -> tuple[int, str, bytes]:
    return status, "application/json; charset=utf-8", json.dumps(payload, indent=2).encode("utf-8")


def _loads(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _first(query: dict[str, list[str]], name: str) -> str:
    values = query.get(name) or []
    return values[0] if values else ""


def _int_query(query: dict[str, list[str]], name: str, default: int) -> int:
    try:
        return int(_first(query, name) or default)
    except ValueError:
        return default


def auto_import_benchmark_reports(store_root: Path, store: LocalTraceStore) -> int:
    workspace = _workspace_from_store_root(store_root)
    reports_root = workspace / ".mikucli" / "evaluation" / "bench" / "runs"
    if not reports_root.is_dir():
        return 0
    imported = 0
    for report in sorted(reports_root.glob("*.json")):
        try:
            store.import_eval_report(report)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        imported += 1
    return imported


def _workspace_from_store_root(store_root: Path) -> Path:
    resolved = store_root.resolve()
    if resolved.name == "observability" and resolved.parent.name == ".mikucli":
        return resolved.parent.parent
    return Path.cwd()


def _dashboard_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>mikucli observability</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 24px; color: #202124; background: #f7f7f4; }
    table { border-collapse: collapse; width: 100%; background: white; }
    th, td { border: 1px solid #d8d8d0; padding: 8px; text-align: left; vertical-align: top; }
    th { background: #ecece4; }
    button { padding: 6px 10px; }
    .grid { display: grid; grid-template-columns: 1fr; gap: 20px; }
    pre { white-space: pre-wrap; background: #202124; color: #f1f3f4; padding: 12px; overflow: auto; }
    .empty { padding: 12px; background: #fff8d7; border: 1px solid #e0c55f; margin: 8px 0; }
  </style>
</head>
<body>
  <h1>mikucli observability</h1>
  <div class="grid">
    <section><h2>Runs</h2><table id="runs"></table></section>
    <section><h2>Cases</h2><table id="cases"></table></section>
    <section><h2>Trace</h2><table id="spans"></table></section>
    <section><h2>Details</h2><pre id="details">{}</pre></section>
  </div>
  <script>
    const details = document.querySelector('#details');
    const table = (id, headers, rows, emptyMessage = 'No rows.') => {
      const target = document.querySelector(id);
      if (!rows.length) {
        target.innerHTML = `<tr><td class="empty" colspan="${headers.length}">${emptyMessage}</td></tr>`;
        return;
      }
      target.innerHTML = '<tr>' + headers.map(h => `<th>${h}</th>`).join('') + '</tr>' +
        rows.map(row => '<tr>' + row.map(cell => `<td>${cell ?? ''}</td>`).join('') + '</tr>').join('');
    };
    const show = value => details.textContent = JSON.stringify(value, null, 2);
    async function loadRuns() {
      const payload = await fetch('/runs').then(r => r.json());
      table('#runs', ['Run', 'Model', 'Success', 'Cases', 'Traces'], payload.runs.map(run => [
        `<button onclick="loadCases('${run.run_id}')">${run.run_id}</button>`,
        run.model,
        `${(run.success_rate * 100).toFixed(1)}%`,
        `${run.passed_cases}/${run.total_cases}`,
        run.traced_cases ? `${run.traced_cases}/${run.total_cases}` : 'none'
      ]), 'No imported eval runs. Run an eval or restart the dashboard from the workspace root so it can auto-import .mikucli/evaluation/bench/runs/*.json.');
      show(payload);
      const firstTracedRun = payload.runs.find(run => run.traced_cases > 0);
      if (firstTracedRun) {
        loadCases(firstTracedRun.run_id);
      } else if (payload.runs.length) {
        loadCases(payload.runs[0].run_id);
      }
    }
    async function loadCases(runId) {
      const payload = await fetch(`/runs/${runId}/cases`).then(r => r.json());
      table('#cases', ['Status', 'Case', 'Trace', 'Latency'], payload.cases.map(item => [
        item.passed ? 'PASS' : 'FAIL',
        `<button onclick="loadCase('${item.case_result_id}')">${item.case_id}</button>`,
        item.trace_id ? `<button onclick="loadSpans('${item.trace_id}')">${item.trace_id.slice(0, 10)}</button>` : 'no trace',
        `${item.metrics.elapsed_seconds ?? 0}s`
      ]), 'No cases for this run.');
      show(payload);
      const firstTracedCase = payload.cases.find(item => item.trace_id);
      if (firstTracedCase) {
        loadSpans(firstTracedCase.trace_id);
      } else if (payload.cases.length) {
        loadCase(payload.cases[0].case_result_id);
        table('#spans', ['Name', 'Parent', 'Status', 'Duration ms'], [], 'This run was imported from an older report and has no trace IDs. Run a new eval to capture trace spans.');
      }
    }
    async function loadCase(caseId) { show(await fetch(`/cases/${caseId}`).then(r => r.json())); }
    async function loadSpans(traceId) {
      const payload = await fetch(`/traces/${traceId}/spans`).then(r => r.json());
      table('#spans', ['Name', 'Parent', 'Status', 'Duration ms'], payload.spans.map(span => [
        span.name, span.parent_span_id || '', span.status, span.duration_ms ?? ''
      ]), 'No trace spans for this case. Older imported reports may not have trace IDs.');
      show(payload);
    }
    loadRuns();
  </script>
</body>
</html>"""


if __name__ == "__main__":
    raise SystemExit(main())
