from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from typing import Any
from unittest.mock import patch

from mikucli.evaluation.bench.runner import BenchmarkRunner
from mikucli.evaluation.bench.tasks import all_benchmark_cases
from mikucli.llm import AssistantMessage, TokenUsage, ToolCall
from mikucli.observability.api import response_for
from mikucli.observability.compare import compare_runs
from mikucli.observability.recorder import LocalTraceRecorder, create_trace_recorder
from mikucli.observability.store import LocalTraceStore
from mikucli.react import AgentSession
from mikucli.tools import ToolRegistry
from mikucli.workspace import Workspace


class FakeClient:
    def __init__(self, responses: list[AssistantMessage]) -> None:
        self.responses = responses

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        stream: bool = False,
    ) -> AssistantMessage:
        return self.responses.pop(0)


class FakeConsole:
    def progress(self, message: str) -> None:
        pass

    def tool_request(self, name: str, arguments: dict[str, Any]) -> None:
        pass

    def tool_result(self, name: str, ok: bool, content: str, diff: str = "") -> None:
        pass

    def answer(self, content: str) -> None:
        pass

    def token_usage(self, usage: TokenUsage) -> None:
        pass


class ObservabilityTests(unittest.TestCase):
    def test_startup_recovery_abandons_stale_trace_and_only_unfinished_spans(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = LocalTraceStore(root / ".mikucli" / "observability", mode="sqlite")
            recorder = LocalTraceRecorder(store, stale_after_seconds=None)
            stale_trace_id = recorder.start_trace(
                run_id="stale-run",
                task_prompt="stale task",
                workspace=str(root),
                model="fake-model",
                session_mode="multi_agent",
            )
            open_span_id = recorder.start_span(trace_id=stale_trace_id, name="llm.chat", kind="llm")
            finished_span_id = recorder.start_span(trace_id=stale_trace_id, name="tool.invoke", kind="tool")
            recorder.end_span(finished_span_id)
            recent_trace_id = recorder.start_trace(
                run_id="recent-run",
                task_prompt="recent task",
                workspace=str(root),
                model="fake-model",
                session_mode="multi_agent",
            )
            recent_span_id = recorder.start_span(trace_id=recent_trace_id, name="agent.session", kind="agent")
            with closing(sqlite3.connect(store.sqlite_path)) as connection:
                connection.execute(
                    "update traces set started_at = '2000-01-01T00:00:00+00:00' where trace_id = ?",
                    (stale_trace_id,),
                )
                connection.execute(
                    "update spans set started_at = '2000-01-01T00:00:00+00:00' where trace_id = ?",
                    (stale_trace_id,),
                )
                finished_before = connection.execute(
                    "select ended_at, status from spans where span_id = ?",
                    (finished_span_id,),
                ).fetchone()
                connection.commit()

            with patch.dict(
                "os.environ",
                {
                    "MIKUCLI_OBS_ENABLED": "1",
                    "MIKUCLI_OBS_STORE": "sqlite",
                    "MIKUCLI_OBS_STALE_AFTER_SECONDS": "3600",
                },
            ):
                create_trace_recorder(root)

            with closing(sqlite3.connect(store.sqlite_path)) as connection:
                stale_trace = connection.execute(
                    "select ended_at, status, attributes_json from traces where trace_id = ?",
                    (stale_trace_id,),
                ).fetchone()
                open_span = connection.execute(
                    "select ended_at, duration_ms, status, attributes_json from spans where span_id = ?",
                    (open_span_id,),
                ).fetchone()
                finished_after = connection.execute(
                    "select ended_at, status from spans where span_id = ?",
                    (finished_span_id,),
                ).fetchone()
                recent_trace = connection.execute(
                    "select ended_at, status from traces where trace_id = ?",
                    (recent_trace_id,),
                ).fetchone()
                recent_span = connection.execute(
                    "select ended_at, status from spans where span_id = ?",
                    (recent_span_id,),
                ).fetchone()

            self.assertEqual(stale_trace[1], "abandoned")
            self.assertIsNotNone(stale_trace[0])
            self.assertEqual(json.loads(stale_trace[2])["recovery.unfinished_span_count"], 1)
            self.assertEqual(open_span[2], "abandoned")
            self.assertIsNotNone(open_span[0])
            self.assertIsNotNone(open_span[1])
            self.assertTrue(json.loads(open_span[3])["recovery.ended_at_estimated"])
            self.assertEqual(finished_after, finished_before)
            self.assertEqual(recent_trace, (None, "running"))
            self.assertEqual(recent_span, (None, "running"))

    def test_recorder_persists_trace_span_event_and_metric(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalTraceStore(Path(tmp), mode="both")
            recorder = LocalTraceRecorder(store)

            trace_id = recorder.start_trace(
                run_id="run-1",
                task_prompt="hello",
                workspace=str(tmp),
                model="fake-model",
                session_mode="single_agent",
            )
            span_id = recorder.start_span(
                trace_id=trace_id,
                name="agent.session",
                kind="agent",
                attributes={"agent.name": "agent"},
            )
            child_id = recorder.start_span(
                trace_id=trace_id,
                name="llm.chat",
                kind="llm",
                parent_span_id=span_id,
            )
            recorder.add_event(child_id, "retry", {"llm.retry_reason": "rate_limit"})
            recorder.record_metric(trace_id, "tokens", 42, unit="tokens")
            recorder.end_span(child_id, attributes={"llm.total_tokens": 42})
            recorder.end_span(span_id)
            recorder.end_trace(trace_id)

            with closing(sqlite3.connect(store.sqlite_path)) as connection:
                trace_count = connection.execute("select count(*) from traces").fetchone()[0]
                spans = connection.execute(
                    "select name, parent_span_id, status, attributes_json from spans order by started_at"
                ).fetchall()
                event_count = connection.execute("select count(*) from span_events").fetchone()[0]
                metric_count = connection.execute("select count(*) from metrics").fetchone()[0]

            self.assertEqual(trace_count, 1)
            self.assertEqual([span[0] for span in spans], ["agent.session", "llm.chat"])
            self.assertIsNone(spans[0][1])
            self.assertEqual(spans[1][1], span_id)
            self.assertEqual(spans[1][2], "ok")
            self.assertEqual(json.loads(spans[1][3])["llm.total_tokens"], 42)
            self.assertEqual(event_count, 1)
            self.assertEqual(metric_count, 1)
            self.assertTrue((store.trace_dir / f"{trace_id}.jsonl").exists())

    def test_single_agent_turn_records_llm_and_tool_spans(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text("hello", encoding="utf-8")
            store = LocalTraceStore(root / ".mikucli" / "observability", mode="sqlite")
            recorder = LocalTraceRecorder(store)
            client = FakeClient(
                [
                    AssistantMessage(
                        content="",
                        tool_calls=[ToolCall(id="call_1", name="list_files", arguments={})],
                        raw={},
                        token_usage=TokenUsage(prompt_tokens=10, completion_tokens=3, total_tokens=13),
                    ),
                    AssistantMessage(
                        content="README.md exists.",
                        tool_calls=[],
                        raw={},
                        token_usage=TokenUsage(prompt_tokens=20, completion_tokens=4, total_tokens=24),
                    ),
                ]
            )
            session = AgentSession(
                client=client,  # type: ignore[arg-type]
                model="fake-model",
                workspace=root,
                tools=ToolRegistry(Workspace(root)),
                console=FakeConsole(),  # type: ignore[arg-type]
                trace_recorder=recorder,
            )

            result = session.run_turn("list files")

            payload = json.loads(result.log_path.read_text(encoding="utf-8"))
            trace_id = payload["metadata"]["trace_id"]
            self.assertEqual(result.final_answer, "README.md exists.")
            self.assertTrue(trace_id)

            with closing(sqlite3.connect(store.sqlite_path)) as connection:
                trace = connection.execute(
                    "select trace_id, run_id, status from traces where trace_id = ?",
                    (trace_id,),
                ).fetchone()
                spans = connection.execute(
                    """
                    select name, parent_span_id, status, attributes_json
                    from spans
                    where trace_id = ?
                    order by started_at
                    """,
                    (trace_id,),
                ).fetchall()

            self.assertEqual(trace[0], trace_id)
            self.assertEqual(trace[2], "ok")
            self.assertEqual([span[0] for span in spans], ["agent.session", "llm.chat", "tool.invoke", "llm.chat"])
            with closing(sqlite3.connect(store.sqlite_path)) as connection:
                agent_span_id = next(
                    row[0]
                    for row in connection.execute(
                        "select span_id from spans where trace_id = ? and name = 'agent.session'",
                        (trace_id,),
                    )
                )
            for name, parent_span_id, status, raw_attributes in spans[1:]:
                self.assertEqual(parent_span_id, agent_span_id)
                self.assertEqual(status, "ok")
                attributes = json.loads(raw_attributes)
                if name == "tool.invoke":
                    self.assertEqual(attributes["tool.name"], "list_files")
                    self.assertTrue(attributes["tool.ok"])
                if name == "llm.chat":
                    self.assertIn("llm.total_tokens", attributes)

    def test_benchmark_result_has_trace_id_and_imported_eval_rows(self) -> None:
        case = _case("repo_inspection:built_in_single_agent")
        client = FakeClient(
            [
                AssistantMessage(
                    content="Acorn Ledger uses src/acorn and tests/test_calculator.py.",
                    tool_calls=[],
                    raw={},
                    token_usage=TokenUsage(total_tokens=5),
                )
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results, result_path, _ = BenchmarkRunner(root=root, client=client, model="fake-model").run([case])
            result = results[0]

            self.assertTrue(result.trace_id)
            self.assertEqual(result.run_group_id, json.loads(result_path.read_text(encoding="utf-8"))["run_id"])
            self.assertTrue(result.hallucination_results)
            self.assertTrue(result.tool_correctness_results)
            store = LocalTraceStore(root / ".mikucli" / "observability", mode="sqlite")
            imported = store.fetch_one("select * from eval_cases where trace_id = ?", (result.trace_id,))
            checks = store.fetch_all("select * from eval_checks where case_result_id = ?", (imported["case_result_id"],))

            self.assertIsNotNone(imported)
            self.assertGreaterEqual(len(checks), 1)

    def test_compare_and_dashboard_api_read_imported_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = LocalTraceStore(root / ".mikucli" / "observability", mode="sqlite")
            base_report = root / "base.json"
            head_report = root / "head.json"
            base_report.write_text(json.dumps(_report("base", passed=True, elapsed=10, tokens=100)), encoding="utf-8")
            head_report.write_text(json.dumps(_report("head", passed=False, elapsed=12, tokens=150)), encoding="utf-8")
            store.import_eval_report(base_report)
            store.import_eval_report(head_report)

            comparison = compare_runs(store, "base", "head")
            status, _, body = response_for("/runs", {}, store)
            failures_status, _, failures_body = response_for("/failures", {"limit": ["10"]}, store)

            self.assertEqual(comparison["cases"][0]["category"], "new_failure")
            self.assertEqual(status, 200)
            self.assertEqual(len(json.loads(body)["runs"]), 2)
            self.assertEqual(failures_status, 200)
            self.assertEqual(json.loads(failures_body)["failures"][0]["case_id"], "case-a")

    def test_dashboard_html_exposes_compare_retries_and_case_signals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalTraceStore(Path(tmp), mode="sqlite")

            status, content_type, body = response_for("/", {}, store)
            html = body.decode("utf-8")

            self.assertEqual(status, 200)
            self.assertIn("text/html", content_type)
            self.assertIn("Regression Comparison", html)
            self.assertIn("Model Retries", html)
            self.assertIn("Tool Calls", html)
            self.assertIn("Case Signals", html)
            self.assertIn("/compare?base=", html)


if __name__ == "__main__":
    unittest.main()


def _case(case_id: str):
    for case in all_benchmark_cases():
        if case.id == case_id:
            return case
    raise AssertionError(f"unknown case: {case_id}")


def _report(run_id: str, *, passed: bool, elapsed: float, tokens: int) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "run_group_id": run_id,
        "started_at": "2026-01-01T00:00:00+00:00",
        "model": "fake-model",
        "summary": {
            "total_cases": 1,
            "passed_cases": 1 if passed else 0,
            "success_rate": 1.0 if passed else 0.0,
        },
        "results": [
            {
                "case_id": "case-a",
                "task_id": "task-a",
                "session_mode": "built_in_single_agent",
                "passed": passed,
                "trace_id": f"trace-{run_id}",
                "workspace": "workspace",
                "run_log_path": "run-log.json",
                "final_answer": "done",
                "changed_paths": [],
                "metrics": {
                    "tool_call_count": 1,
                    "model_retries": 0,
                    "step_retries": 0,
                    "elapsed_seconds": elapsed,
                    "agent_latency_seconds": 1.0,
                    "llm_latency_seconds": elapsed - 1,
                    "cost": {"total_tokens": tokens},
                },
                "failure_reasons": [] if passed else [{"category": "check_failed", "source": "check", "message": "failed"}],
                "check_results": [{"name": "check", "passed": passed, "messages": [], "category": "task_success", "evidence": {}}],
                "hallucination_results": [],
                "tool_correctness_results": [],
                "tool_calls": [],
                "approvals": [],
            }
        ],
    }
