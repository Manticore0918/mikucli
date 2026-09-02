from __future__ import annotations


def dashboard_html() -> str:
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
    .empty[hidden] { display: none; }
    .controls { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    select { padding: 6px 8px; }
  </style>
</head>
<body>
  <h1>mikucli observability</h1>
  <div class="grid">
    <section><h2>Runs</h2><table id="runs"></table></section>
    <section>
      <h2>Regression Comparison</h2>
      <div class="controls">
        <label>Base <select id="base-run"></select></label>
        <label>Head <select id="head-run"></select></label>
        <button id="compare-runs">Compare</button>
      </div>
      <table id="comparison-context"></table>
      <p id="comparison-warning" class="empty" hidden></p>
      <table id="comparison"></table>
    </section>
    <section><h2>Cases</h2><table id="cases"></table></section>
    <section><h2>Case Signals</h2><table id="signals"></table></section>
    <section><h2>Trace</h2><table id="spans"></table></section>
    <section><h2>Details</h2><pre id="details">{}</pre></section>
  </div>
  <script>
    const details = document.querySelector('#details');
    const table = (id, headers, rows, emptyMessage = 'No rows.') => {
      const target = document.querySelector(id);
      target.replaceChildren();
      if (!rows.length) {
        const row = target.insertRow();
        const cell = row.insertCell();
        cell.className = 'empty';
        cell.colSpan = headers.length;
        cell.textContent = emptyMessage;
        return;
      }
      const header = target.insertRow();
      headers.forEach(value => {
        const cell = document.createElement('th');
        cell.textContent = value;
        header.append(cell);
      });
      rows.forEach(values => {
        const row = target.insertRow();
        values.forEach(value => {
          const cell = row.insertCell();
          if (value instanceof Node) {
            cell.append(value);
          } else {
            cell.textContent = value ?? '';
          }
        });
      });
    };
    const button = (label, action) => {
      const value = document.createElement('button');
      value.textContent = label;
      value.addEventListener('click', action);
      return value;
    };
    const show = value => details.textContent = JSON.stringify(value, null, 2);
    let detailRequest = 0;
    const reveal = selector => {
      const el = document.querySelector(selector);
      if (el) el.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    };
    const metric = (item, name) => item.metrics?.[name] ?? 0;
    const oneDecimal = (value, fallback = '0.0') => {
      if (value === null || value === undefined || value === '') return fallback;
      const numeric = Number(value);
      return Number.isFinite(numeric) ? numeric.toFixed(1) : fallback;
    };
    const integer = (value, fallback = '0') => {
      if (value === null || value === undefined || value === '') return fallback;
      const numeric = Number(value);
      return Number.isFinite(numeric) ? numeric.toFixed(0) : fallback;
    };
    const countFailed = (checks, category) => checks.filter(check => check.category === category && !check.passed).length;
    const showComparisonWarnings = warnings => {
      const target = document.querySelector('#comparison-warning');
      target.textContent = warnings.length ? `Comparison note: ${warnings.join('; ')}` : '';
      target.hidden = !warnings.length;
    };
    const showComparisonContext = payload => {
      const rows = [
        ['Base', payload.base_run],
        ['Head', payload.head_run]
      ].filter(([, run]) => run);
      table('#comparison-context', ['Side', 'Runtime', 'Model', 'Success', 'Cases', 'Started', 'Run'], rows.map(([side, run]) => [
        side,
        run.runtime ?? 'unknown',
        run.model ?? 'unknown',
        `${oneDecimal((run.success_rate ?? 0) * 100)}%`,
        `${run.passed_cases ?? 0}/${run.total_cases ?? 0}`,
        run.started_at ?? 'unknown',
        run.run_id ?? 'unknown'
      ]), 'Choose two runs and compare them to see run details.');
    };
    const populateRunSelects = runs => {
      const base = document.querySelector('#base-run');
      const head = document.querySelector('#head-run');
      base.replaceChildren(...runs.map(run => new Option(run.run_id, run.run_id)));
      head.replaceChildren(...runs.map(run => new Option(run.run_id, run.run_id)));
      if (runs.length > 1) {
        document.querySelector('#base-run').value = runs[1].run_id;
        document.querySelector('#head-run').value = runs[0].run_id;
      }
    };
    async function loadRuns() {
      const payload = await fetch('/runs').then(r => r.json());
      populateRunSelects(payload.runs);
      table('#runs', ['Run', 'Model', 'Success', 'Cases', 'Traces', 'Tool Calls', 'Model Retries', 'Step Retries'], payload.runs.map(run => [
        button(run.run_id, () => loadCases(run.run_id, { focus: true })),
        run.model,
        `${oneDecimal(run.success_rate * 100)}%`,
        `${run.passed_cases}/${run.total_cases}`,
        run.traced_cases ? `${run.traced_cases}/${run.total_cases}` : 'none',
        integer(run.summary?.tool_call_count),
        integer(run.summary?.model_retries),
        integer(run.summary?.step_retries)
      ]), 'No imported eval runs. Run an eval or restart the dashboard from the workspace root so it can auto-import .mikucli/evaluation/bench/runs/*.json.');
      show(payload);
      const firstTracedRun = payload.runs.find(run => run.traced_cases > 0);
      if (firstTracedRun) {
        loadCases(firstTracedRun.run_id);
      } else if (payload.runs.length) {
        loadCases(payload.runs[0].run_id);
      }
    }
    async function loadCases(runId, { focus = false } = {}) {
      const payload = await fetch(`/runs/${encodeURIComponent(runId)}/cases`).then(r => r.json());
      table('#cases', ['Status', 'Case', 'Trace', 'Latency', 'Tool Calls', 'Model Retries', 'Step Retries', 'Tokens'], payload.cases.map(item => [
        item.passed ? 'PASS' : 'FAIL',
        button(item.case_id, () => loadCase(item.case_result_id, { focus: true })),
        item.trace_id ? button(item.trace_id.slice(0, 10), () => loadSpans(item.trace_id, { focus: true })) : 'no trace',
        `${oneDecimal(metric(item, 'elapsed_seconds'))}s`,
        integer(metric(item, 'tool_call_count')),
        integer(metric(item, 'model_retries')),
        integer(metric(item, 'step_retries')),
        integer(item.metrics?.cost?.total_tokens, 'unknown')
      ]), 'No cases for this run.');
      show(payload);
      const firstTracedCase = payload.cases.find(item => item.trace_id);
      if (firstTracedCase) {
        loadSpans(firstTracedCase.trace_id);
      } else {
        table('#spans', ['Name', 'Parent', 'Status', 'Duration ms'], [], 'This run was imported from an older report and has no trace IDs. Run a new eval to capture trace spans.');
      }
      if (payload.cases.length) {
        loadCase(payload.cases[0].case_result_id);
      }
      if (focus) reveal('#cases');
    }
    async function loadCase(caseId, { focus = false } = {}) {
      const request = ++detailRequest;
      const payload = await fetch(`/cases/${encodeURIComponent(caseId)}`).then(r => r.json());
      if (request !== detailRequest) return;
      const checks = payload.checks ?? [];
      table('#signals', ['Category', 'Check', 'Status', 'Messages'], checks.map(check => [
        check.category,
        check.name,
        check.passed ? 'PASS' : 'FAIL',
        (check.messages ?? []).join('; ')
      ]), 'No structured checks for this case.');
      const toolFailures = (payload.tool_calls ?? []).filter(call => !call.ok);
      if (toolFailures.length) {
        table('#signals', ['Category', 'Check', 'Status', 'Messages'], [
          ...checks.map(check => [check.category, check.name, check.passed ? 'PASS' : 'FAIL', (check.messages ?? []).join('; ')]),
          ...toolFailures.map(call => ['tool_call', call.name, 'FAIL', call.content])
        ]);
      }
      show({
        summary: {
          case_id: payload.case?.case_id,
          passed: payload.case?.passed,
          tool_correctness_failures: countFailed(checks, 'tool_correctness'),
          failed_tool_calls: toolFailures.length
        },
        ...payload
      });
      if (focus) reveal('#signals');
    }
    async function loadSpans(traceId, { focus = false } = {}) {
      const request = ++detailRequest;
      const payload = await fetch(`/traces/${encodeURIComponent(traceId)}/spans`).then(r => r.json());
      table('#spans', ['Name', 'Parent', 'Status', 'Duration ms'], payload.spans.map(span => [
        span.name, span.parent_span_id || '', span.status, oneDecimal(span.duration_ms, '')
      ]), 'No trace spans for this case. Older imported reports may not have trace IDs.');
      if (request !== detailRequest) return;
      show(payload);
      if (focus) reveal('#spans');
    }
    async function loadComparison() {
      const base = document.querySelector('#base-run').value;
      const head = document.querySelector('#head-run').value;
      if (!base || !head) {
        showComparisonContext({});
        showComparisonWarnings([]);
        table('#comparison', ['Case', 'Category', 'Pass Change', 'Latency Delta', 'Token Delta', 'Retry Delta'], [], 'Need at least two runs to compare.');
        return;
      }
      const payload = await fetch(`/compare?base=${encodeURIComponent(base)}&head=${encodeURIComponent(head)}`).then(r => r.json());
      showComparisonContext(payload);
      showComparisonWarnings(payload.compatibility_warnings ?? []);
      if (!payload.compatible) {
        table('#comparison', ['Case', 'Category', 'Pass Change', 'Latency Delta', 'Token Delta', 'Retry Delta'], [],
          `Runs are not comparable: ${(payload.compatibility_warnings ?? []).join('; ')}`);
        show(payload);
        return;
      }
      table('#comparison', ['Case', 'Category', 'Pass Change', 'Latency Delta', 'Token Delta', 'Retry Delta'], (payload.cases ?? []).map(item => [
        item.case_id,
        item.category,
        `${item.details?.base_passed ?? ''} -> ${item.details?.head_passed ?? ''}`,
        oneDecimal(item.details?.elapsed_seconds_delta),
        integer(item.details?.total_token_delta, 'unknown'),
        `model ${integer(item.details?.model_retry_delta)}, step ${integer(item.details?.step_retry_delta)}`
      ]), 'No comparison rows.');
      show(payload);
    }
    document.querySelector('#compare-runs').addEventListener('click', loadComparison);
    loadRuns();
  </script>
</body>
</html>"""
