from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
import tempfile
import threading
import unittest
from contextlib import closing
from typing import Any

from mikucli.codebase.types import SearchResult
from mikucli.llm import AssistantMessage, TokenUsage, ToolCall
from mikucli.multi_agent import (
    DEFAULT_SUBAGENTS,
    OrchestratorSession,
    ReadOnlyTools,
    SerializedMutationTools,
    parse_execution_plan,
    parse_review_decision,
)
from mikucli.observability.recorder import LocalTraceRecorder
from mikucli.observability.store import LocalTraceStore
from mikucli.skills import Skill, SkillScope
from mikucli.tools import ToolRegistry, ToolResult
from mikucli.workspace import Workspace


class RoutingFakeClient:
    def __init__(self, plan: dict[str, Any], failing_steps: set[str] | None = None) -> None:
        self.plan = plan
        self.failing_steps = failing_steps or set()
        self.requests: list[list[dict[str, Any]]] = []
        self.tool_requests: list[tuple[str, list[dict[str, Any]]]] = []
        self.worker_attempts: dict[str, int] = {}
        self.lock = threading.Lock()

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        stream: bool = False,
    ) -> AssistantMessage:
        with self.lock:
            self.requests.append(messages)
        system = str(messages[0]["content"])
        user = str(messages[-1]["content"])
        if "planner subagent" in system:
            with self.lock:
                self.tool_requests.append(("planner", tools))
            return _message(json.dumps(self.plan))
        if "worker subagent" in system:
            with self.lock:
                self.tool_requests.append(("worker", tools))
            step_id = _step_id_from_prompt(user)
            with self.lock:
                attempt = self.worker_attempts.get(step_id, 0) + 1
                self.worker_attempts[step_id] = attempt
            if step_id in self.failing_steps:
                return _message(f"{step_id} incomplete on attempt {attempt}.")
            return _message(f"{step_id} completed on attempt {attempt}.")
        if "reviewer subagent" in system:
            with self.lock:
                self.tool_requests.append(("reviewer", tools))
            step_id = _step_id_from_prompt(user)
            if step_id in self.failing_steps:
                return _message(
                    json.dumps(
                        {
                            "approved": False,
                            "summary": f"{step_id} rejected.",
                            "issues": [f"{step_id} needs more work."],
                            "suggestions": [f"Rerun {step_id}."],
                        }
                    )
                )
            return _message(
                json.dumps(
                    {
                        "approved": True,
                        "summary": f"{step_id} passed review.",
                        "issues": [],
                        "suggestions": [],
                    }
                )
            )
        return _message("unexpected request")


class ReviewerToolThenApproveClient(RoutingFakeClient):
    def __init__(self, plan: dict[str, Any]) -> None:
        super().__init__(plan)
        self.review_tool_requested = False

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        stream: bool = False,
    ) -> AssistantMessage:
        system = str(messages[0]["content"])
        if "reviewer subagent" in system and not self.review_tool_requested:
            self.review_tool_requested = True
            with self.lock:
                self.requests.append(messages)
                self.tool_requests.append(("reviewer", tools))
            return AssistantMessage(
                content="",
                tool_calls=[ToolCall(id="call_1", name="read_file", arguments={"path": "demo/pom.xml"})],
                raw={},
                token_usage=TokenUsage(total_tokens=1),
            )
        if "reviewer subagent" in system and self.review_tool_requested:
            with self.lock:
                self.requests.append(messages)
                self.tool_requests.append(("reviewer", tools))
            return _message(
                json.dumps(
                    {
                        "approved": True,
                        "summary": "step-1 passed review.",
                        "issues": [],
                        "suggestions": [],
                    }
                )
            )
        return super().chat(model=model, messages=messages, tools=tools, stream=stream)


class DirectPlannerAnswerClient(RoutingFakeClient):
    def __init__(self, answer: str) -> None:
        super().__init__({"steps": [{"id": "unused", "task": "unused"}]})
        self.answer = answer

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        stream: bool = False,
    ) -> AssistantMessage:
        with self.lock:
            self.requests.append(messages)
        system = str(messages[0]["content"])
        if "planner subagent" in system:
            return _message(self.answer)
        return _message("unexpected request")


class SimulatedInterruption(BaseException):
    pass


class InterruptingWorkerClient(RoutingFakeClient):
    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        stream: bool = False,
    ) -> AssistantMessage:
        if "worker subagent" in str(messages[0]["content"]):
            raise SimulatedInterruption("worker interrupted")
        return super().chat(model=model, messages=messages, tools=tools, stream=stream)


class FakeConsole:
    def __init__(self) -> None:
        self.answers: list[str] = []
        self.progress_messages: list[str] = []

    def progress(self, message: str) -> None:
        self.progress_messages.append(message)

    def tool_request(self, name: str, arguments: dict[str, Any]) -> None:
        pass

    def tool_result(self, name: str, ok: bool, content: str, diff: str = "") -> None:
        pass

    def answer(self, content: str) -> None:
        self.answers.append(content)

    def token_usage(self, usage: TokenUsage) -> None:
        pass


class FakeCodebaseService:
    def search(self, query: str, limit: int = 8) -> list[SearchResult]:
        return [
            SearchResult(
                path="README.md",
                start_line=1,
                end_line=1,
                kind="text",
                symbol="",
                content="mikucli",
                hybrid_score=0.1,
            )
        ]


class FakeMcpLikeTools:
    def schemas(self) -> list[dict[str, Any]]:
        return [
            {"type": "function", "function": {"name": "read_github_file", "description": "", "parameters": {}}},
            {"type": "function", "function": {"name": "write_github_file", "description": "", "parameters": {}}},
        ]

    def read_only_tool_names(self) -> set[str]:
        return {"read_github_file"}

    def requires_approval(self, name: str, arguments: dict[str, Any] | None = None) -> bool:
        return False

    def invoke(self, name: str, arguments: dict[str, Any]) -> Any:
        return ToolResult(ok=True, content=f"called {name}")


class ConcurrencyTrackingTools:
    def __init__(
        self,
        read_only_names: set[str] | None = None,
        approval_names: set[str] | None = None,
    ) -> None:
        self._read_only_names = read_only_names or set()
        self._approval_names = approval_names or set()
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()
        self.read_barrier = threading.Barrier(2)

    def schemas(self) -> list[dict[str, Any]]:
        return []

    def read_only_tool_names(self) -> set[str]:
        return set(self._read_only_names)

    def requires_approval(self, name: str, arguments: dict[str, Any] | None = None) -> bool:
        return name in self._approval_names

    def invoke(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if name in self._read_only_names and name not in self._approval_names:
                self.read_barrier.wait(timeout=1)
            else:
                time.sleep(0.05)
            return ToolResult(ok=True, content=f"called {name}")
        finally:
            with self.lock:
                self.active -= 1


class MultiAgentTests(unittest.TestCase):
    def test_active_skill_reaches_planner_worker_and_reviewer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = RoutingFakeClient(
                {"steps": [{"id": "step-1", "title": "Inspect", "task": "Inspect auth", "depends_on": []}]}
            )
            session = OrchestratorSession(
                client=client,  # type: ignore[arg-type]
                model="test-model",
                workspace=root,
                tools=ToolRegistry(Workspace(root)),
                console=FakeConsole(),
            )
            skill = Skill(
                name="review",
                description="Review code.",
                instructions="Use the active review workflow.",
                scope=SkillScope.WORKSPACE,
                path=root / ".mikucli" / "skills" / "review" / "SKILL.md",
                content_hash="b" * 64,
                metadata={"name": "review", "description": "Review code."},
            )

            result = session.run_turn("inspect auth", active_skill=skill)

            role_system_prompts = [str(messages[0]["content"]) for messages in client.requests]
            self.assertTrue(any("planner subagent" in prompt for prompt in role_system_prompts))
            self.assertTrue(any("worker subagent" in prompt for prompt in role_system_prompts))
            self.assertTrue(any("reviewer subagent" in prompt for prompt in role_system_prompts))
            self.assertTrue(all("Active Skill: $review" in prompt for prompt in role_system_prompts))
            payload = json.loads(result.log_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["metadata"]["skill"]["content_hash"], "b" * 64)

    def test_default_roster_initializes_one_planner_two_workers_and_one_reviewer(self) -> None:
        roles = [spec.role for spec in DEFAULT_SUBAGENTS]

        self.assertEqual(roles.count("planner"), 1)
        self.assertEqual(roles.count("worker"), 2)
        self.assertEqual(roles.count("reviewer"), 1)
        self.assertEqual([spec.id for spec in DEFAULT_SUBAGENTS], ["planner-1", "worker-1", "worker-2", "reviewer-1"])

    def test_parse_execution_plan_builds_dependency_relations(self) -> None:
        steps = parse_execution_plan(
            json.dumps(
                {
                    "steps": [
                        {"id": "step-1", "task": "First"},
                        {"id": "step-2", "task": "Second", "depends_on": ["step-1"]},
                    ]
                }
            )
        )

        self.assertEqual([step.id for step in steps], ["step-1", "step-2"])
        self.assertEqual(steps[1].depends_on, ["step-1"])

    def test_orchestrator_accepts_direct_planner_answer_for_read_only_task(self) -> None:
        answer = "The sentinel is ORCHID-917 and it names the amber queue."
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            console = FakeConsole()
            session = OrchestratorSession(
                client=DirectPlannerAnswerClient(answer),  # type: ignore[arg-type]
                model="test-model",
                workspace=root,
                tools=ToolRegistry(Workspace(root)),
                console=console,
            )

            result = session.run_turn("Find the sentinel.")

            self.assertEqual(result.final_answer, answer)
            self.assertEqual(console.answers, [answer])

    def test_parse_review_decision_handles_issues_and_suggestions(self) -> None:
        decision = parse_review_decision(
            '{"approved":false,"summary":"not done","issues":["retry"],"suggestions":["fix it"]}'
        )

        self.assertFalse(decision.approved)
        self.assertEqual(decision.summary, "not done")
        self.assertEqual(decision.issues, ["retry"])
        self.assertEqual(decision.suggestions, ["fix it"])
        self.assertEqual(decision.problems, "Issues: retry\nSuggestions: fix it")

    def test_parse_review_decision_keeps_legacy_passed_feedback_fields(self) -> None:
        decision = parse_review_decision('{"passed":"false","feedback":"retry","summary":"not done"}')

        self.assertFalse(decision.approved)
        self.assertEqual(decision.problems, "Issues: retry")

    def test_review_prompt_contains_step_description_and_worker_result(self) -> None:
        plan = {"steps": [{"id": "step-1", "task": "Check the target behavior."}]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = RoutingFakeClient(plan)
            session = OrchestratorSession(
                client=client,  # type: ignore[arg-type]
                model="test-model",
                workspace=root,
                tools=ToolRegistry(Workspace(root)),
                console=FakeConsole(),
            )

            session.run_turn("Do the work.")

            reviewer_prompts = [
                str(messages[-1]["content"])
                for messages in client.requests
                if "reviewer subagent" in str(messages[0]["content"])
            ]
            self.assertEqual(len(reviewer_prompts), 1)
            self.assertIn("Check the target behavior.", reviewer_prompts[0])
            self.assertIn("step-1 completed on attempt 1.", reviewer_prompts[0])

    def test_worker_prompt_includes_completed_dependency_context(self) -> None:
        plan = {
            "steps": [
                {"id": "step-1", "task": "Prepare context."},
                {"id": "step-2", "task": "Use context.", "depends_on": ["step-1"]},
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = RoutingFakeClient(plan)
            session = OrchestratorSession(
                client=client,  # type: ignore[arg-type]
                model="test-model",
                workspace=root,
                tools=ToolRegistry(Workspace(root)),
                console=FakeConsole(),
            )

            session.run_turn("Do dependency work.")

            worker_prompts = [
                str(messages[-1]["content"])
                for messages in client.requests
                if "worker subagent" in str(messages[0]["content"])
            ]
            self.assertIn("Completed dependency step context", worker_prompts[1])
            self.assertIn("step-1: Prepare context.", worker_prompts[1])
            self.assertIn("Result: step-1 completed on attempt 1.", worker_prompts[1])

    def test_dependency_context_is_limited_to_first_500_characters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = OrchestratorSession(
                client=RoutingFakeClient({"steps": [{"id": "step-1", "task": "Do work."}]}),  # type: ignore[arg-type]
                model="test-model",
                workspace=root,
                tools=ToolRegistry(Workspace(root)),
                console=FakeConsole(),
            )
            long_result = "x" * 600
            dependency = parse_execution_plan(
                json.dumps(
                    {
                        "steps": [
                            {"id": "step-1", "task": "Prepare context."},
                            {"id": "step-2", "task": "Use context.", "depends_on": ["step-1"]},
                        ]
                    }
                )
            )
            dependency[0].status = "passed"
            dependency[0].result = long_result
            session._current_step_by_id = {step.id: step for step in dependency}

            context = session._dependency_context(dependency[1])

            self.assertEqual(len(context), 500)
            self.assertEqual(context, ("step-1: Prepare context.\nResult: " + long_result)[:500])

    def test_orchestrator_runs_planner_workers_reviewer_and_summarizes_to_memory(self) -> None:
        plan = {
            "steps": [
                {"id": "step-1", "title": "Inspect", "task": "Inspect files."},
                {"id": "step-2", "title": "Implement", "task": "Implement change.", "depends_on": ["step-1"]},
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = RoutingFakeClient(plan)
            console = FakeConsole()
            session = OrchestratorSession(
                client=client,  # type: ignore[arg-type]
                model="test-model",
                workspace=root,
                tools=ToolRegistry(Workspace(root)),
                console=console,
            )

            result = session.run_turn("Do the work.")

            self.assertIn("step-1 [passed]", result.final_answer)
            self.assertIn("step-2 [passed]", result.final_answer)
            self.assertIn("step-1 completed", result.final_answer)
            self.assertIn("step-2 completed", result.final_answer)
            self.assertTrue(any(entry.content == result.final_answer for entry in session.memory.active_entries))
            self.assert_chat_history_empty(session.subagents["worker-1"])
            self.assert_chat_history_empty(session.subagents["worker-2"])
            self.assert_chat_history_empty(session.subagents["reviewer-1"])
            self.assertTrue(
                all(
                    _tool_names(tools) == ["list_files", "read_file"]
                    for role, tools in client.tool_requests
                    if role in {"planner", "reviewer"}
                )
            )
            self.assertTrue(all(tools != [] for role, tools in client.tool_requests if role == "worker"))
            self.assertIn("phase 1: planning", console.progress_messages)
            self.assertIn("plan:", console.progress_messages)
            self.assertIn("step-1: Inspect", console.progress_messages)
            self.assertIn("step-2: Implement", console.progress_messages)
            self.assertIn("phase 2: executing", console.progress_messages)
            self.assertIn("worker-1 executing [step-1]: Inspect", console.progress_messages)
            self.assertIn("reviewer reviewing the results of [step-1]", console.progress_messages)
            self.assertIn("[step-1] review approved", console.progress_messages)

    def test_planner_and_reviewer_receive_only_read_only_tools(self) -> None:
        plan = {"steps": [{"id": "step-1", "task": "Do work."}]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = OrchestratorSession(
                client=RoutingFakeClient(plan),  # type: ignore[arg-type]
                model="test-model",
                workspace=root,
                tools=ToolRegistry(Workspace(root)),
                console=FakeConsole(),
            )

            self.assertEqual(_tool_names(session.subagents["planner-1"].tools.schemas()), ["list_files", "read_file"])
            self.assertEqual(_tool_names(session.subagents["reviewer-1"].tools.schemas()), ["list_files", "read_file"])
            self.assertIn("write_file", _tool_names(session.subagents["worker-1"].tools.schemas()))
            self.assertIn("run_shell", _tool_names(session.subagents["worker-2"].tools.schemas()))
            denied = session.subagents["reviewer-1"].tools.invoke("write_file", {"path": "x.txt", "content": "x"})
            self.assertFalse(denied.ok)
            self.assertIn("not available", denied.content)

    def test_team_workers_share_serialized_mutation_tools(self) -> None:
        plan = {"steps": [{"id": "step-1", "task": "Do work."}]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = OrchestratorSession(
                client=RoutingFakeClient(plan),  # type: ignore[arg-type]
                model="test-model",
                workspace=root,
                tools=ToolRegistry(Workspace(root)),
                console=FakeConsole(),
            )

            worker_1_tools = session.subagents["worker-1"].tools
            worker_2_tools = session.subagents["worker-2"].tools

            self.assertIsInstance(worker_1_tools, SerializedMutationTools)
            self.assertIs(worker_1_tools, worker_2_tools)
            self.assertIs(session.subagents["planner-1"].tools.base_tools, worker_1_tools)  # type: ignore[attr-defined]

    def test_serialized_mutation_tools_lock_entire_mutating_invocation(self) -> None:
        base_tools = ConcurrencyTrackingTools()
        tools = SerializedMutationTools(base_tools)
        start = threading.Barrier(3)

        def invoke(path: str) -> None:
            start.wait(timeout=1)
            tools.invoke("write_file", {"path": path, "content": path})

        threads = [
            threading.Thread(target=invoke, args=("one.txt",)),
            threading.Thread(target=invoke, args=("two.txt",)),
        ]
        for thread in threads:
            thread.start()
        start.wait(timeout=1)
        for thread in threads:
            thread.join(timeout=2)

        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(base_tools.max_active, 1)

    def test_serialized_mutation_tools_allow_concurrent_read_only_invocations(self) -> None:
        base_tools = ConcurrencyTrackingTools(read_only_names={"read_file"})
        tools = SerializedMutationTools(base_tools)

        threads = [
            threading.Thread(target=tools.invoke, args=("read_file", {"path": "one.txt"})),
            threading.Thread(target=tools.invoke, args=("read_file", {"path": "two.txt"})),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2)

        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(base_tools.max_active, 2)

    def test_serialized_mutation_tools_serialize_approval_requiring_read_only_invocations(self) -> None:
        base_tools = ConcurrencyTrackingTools(
            read_only_names={"read_remote"},
            approval_names={"read_remote"},
        )
        tools = SerializedMutationTools(base_tools)

        threads = [
            threading.Thread(target=tools.invoke, args=("read_remote", {"id": "one"})),
            threading.Thread(target=tools.invoke, args=("read_remote", {"id": "two"})),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2)

        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(base_tools.max_active, 1)

    def test_planner_and_reviewer_can_use_search_codebase_when_available(self) -> None:
        plan = {"steps": [{"id": "step-1", "task": "Do work."}]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = OrchestratorSession(
                client=RoutingFakeClient(plan),  # type: ignore[arg-type]
                model="test-model",
                workspace=root,
                tools=ToolRegistry(
                    Workspace(root),
                    codebase_service=FakeCodebaseService(),
                ),
                console=FakeConsole(),
            )

            planner_tools = _tool_names(session.subagents["planner-1"].tools.schemas())
            reviewer_tools = _tool_names(session.subagents["reviewer-1"].tools.schemas())
            self.assertEqual(planner_tools, ["list_files", "read_file", "search_codebase"])
            self.assertEqual(reviewer_tools, ["list_files", "read_file", "search_codebase"])
            result = session.subagents["reviewer-1"].tools.invoke("search_codebase", {"query": "what is mikucli?"})
            self.assertTrue(result.ok)
            self.assertIn("README.md:1-1", result.content)

    def test_read_only_tools_uses_active_tool_set_read_only_names(self) -> None:
        tools = ReadOnlyTools(FakeMcpLikeTools())  # type: ignore[arg-type]

        self.assertEqual(_tool_names(tools.schemas()), ["read_github_file"])
        allowed = tools.invoke("read_github_file", {})
        denied = tools.invoke("write_github_file", {})

        self.assertTrue(allowed.ok)
        self.assertFalse(denied.ok)
        self.assertIn("not available", denied.content)

    def test_reviewer_read_file_attempt_does_not_crash_step(self) -> None:
        plan = {"steps": [{"id": "step-1", "task": "Do work."}]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = ReviewerToolThenApproveClient(plan)
            session = OrchestratorSession(
                client=client,  # type: ignore[arg-type]
                model="test-model",
                workspace=root,
                tools=ToolRegistry(Workspace(root)),
                console=FakeConsole(),
            )

            result = session.run_turn("Do the work.")

            self.assertIn("step-1 [passed]", result.final_answer)
            self.assertTrue(client.review_tool_requested)

    def test_failed_step_skips_dependents(self) -> None:
        plan = {
            "steps": [
                {"id": "step-1", "task": "This will fail."},
                {"id": "step-2", "task": "Blocked work.", "depends_on": ["step-1"]},
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = OrchestratorSession(
                client=RoutingFakeClient(plan, failing_steps={"step-1"}),  # type: ignore[arg-type]
                model="test-model",
                workspace=root,
                tools=ToolRegistry(Workspace(root)),
                console=FakeConsole(),
                max_step_attempts=2,
            )

            result = session.run_turn("Do the work.")

            self.assertIn("step-1 [failed]", result.final_answer)
            self.assertIn("step-2 [skipped]", result.final_answer)
            self.assertIn("Blocked by failed or skipped dependencies: step-1", result.final_answer)
            self.assert_chat_history_empty(session.subagents["worker-1"])
            self.assert_chat_history_empty(session.subagents["reviewer-1"])

    def test_independent_steps_are_distributed_to_workers(self) -> None:
        plan = {
            "steps": [
                {"id": "step-1", "task": "First independent task."},
                {"id": "step-2", "task": "Second independent task."},
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = OrchestratorSession(
                client=RoutingFakeClient(plan),  # type: ignore[arg-type]
                model="test-model",
                workspace=root,
                tools=ToolRegistry(Workspace(root)),
                console=FakeConsole(),
            )

            result = session.run_turn("Do independent work.")

            self.assertIn("step-1 [passed] via worker-1", result.final_answer)
            self.assertIn("step-2 [passed] via worker-2", result.final_answer)

    def test_orchestrator_records_workflow_step_and_subagent_spans(self) -> None:
        plan = {"steps": [{"id": "step-1", "task": "Do work."}]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = LocalTraceStore(root / ".mikucli" / "observability", mode="sqlite")
            session = OrchestratorSession(
                client=RoutingFakeClient(plan),  # type: ignore[arg-type]
                model="test-model",
                workspace=root,
                tools=ToolRegistry(Workspace(root)),
                console=FakeConsole(),
                trace_recorder=LocalTraceRecorder(store),
            )

            result = session.run_turn("Do traced work.")

            payload = json.loads(result.log_path.read_text(encoding="utf-8"))
            trace_id = payload["metadata"]["trace_id"]
            with closing(sqlite3.connect(store.sqlite_path)) as connection:
                names = [
                    row[0]
                    for row in connection.execute(
                        "select name from spans where trace_id = ? order by started_at",
                        (trace_id,),
                    )
                ]
                subagent_attrs = [
                    json.loads(row[0])
                    for row in connection.execute(
                        "select attributes_json from spans where trace_id = ? and name = 'subagent.turn'",
                        (trace_id,),
                    )
                ]

            self.assertIn("orchestrator.workflow", names)
            self.assertIn("orchestrator.plan", names)
            self.assertIn("orchestrator.step", names)
            self.assertGreaterEqual(names.count("subagent.turn"), 3)
            self.assertEqual(
                {"planner", "worker", "reviewer"},
                {attrs["subagent.role"] for attrs in subagent_attrs},
            )

    def test_interrupted_orchestrator_closes_trace_and_active_spans(self) -> None:
        plan = {"steps": [{"id": "step-1", "task": "Do work."}]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = LocalTraceStore(root / ".mikucli" / "observability", mode="sqlite")
            session = OrchestratorSession(
                client=InterruptingWorkerClient(plan),  # type: ignore[arg-type]
                model="test-model",
                workspace=root,
                tools=ToolRegistry(Workspace(root)),
                console=FakeConsole(),
                trace_recorder=LocalTraceRecorder(store),
            )

            with self.assertRaises(SimulatedInterruption):
                session.run_turn("Interrupt traced work.")

            with closing(sqlite3.connect(store.sqlite_path)) as connection:
                trace = connection.execute(
                    "select status, ended_at, attributes_json from traces where session_mode = 'multi_agent'"
                ).fetchone()
                spans = connection.execute(
                    "select name, status, ended_at from spans order by started_at"
                ).fetchall()

            self.assertIsNotNone(trace)
            assert trace is not None
            self.assertEqual(trace[0], "error")
            self.assertIsNotNone(trace[1])
            self.assertEqual(json.loads(trace[2])["error.type"], "SimulatedInterruption")
            self.assertTrue(spans)
            self.assertTrue(all(ended_at is not None for _, _, ended_at in spans))
            self.assertIn(("llm.chat", "error"), [(name, status) for name, status, _ in spans])
            self.assertIn(("subagent.turn", "error"), [(name, status) for name, status, _ in spans])
            self.assertIn(("orchestrator.step", "error"), [(name, status) for name, status, _ in spans])
            self.assertIn(("orchestrator.workflow", "error"), [(name, status) for name, status, _ in spans])
            self.assertIn(("agent.session", "error"), [(name, status) for name, status, _ in spans])

    def assert_chat_history_empty(self, agent: Any) -> None:
        self.assertEqual(agent.memory.active_entries, [])
        self.assertEqual(agent.memory.old_entries, [])
        self.assertEqual(agent.memory.summary_entries, [])


def _message(content: str) -> AssistantMessage:
    return AssistantMessage(
        content=content,
        tool_calls=[],
        raw={},
        token_usage=TokenUsage(total_tokens=1),
    )


def _step_id_from_prompt(prompt: str) -> str:
    marker = "ExecutionStep "
    start = prompt.index(marker) + len(marker)
    return prompt[start:].split(":", 1)[0].strip()


def _tool_names(schemas: list[dict[str, Any]]) -> list[str]:
    return [schema["function"]["name"] for schema in schemas]


if __name__ == "__main__":
    unittest.main()
