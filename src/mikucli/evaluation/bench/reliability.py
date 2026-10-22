"""Execute deterministic workflow evals without credentials or external services."""

from __future__ import annotations

from mikucli.tools import ToolRegistry
from mikucli.workspace import Workspace

from .models import ApprovalRecord
from .reliability_scenarios import make_scenario
from .reliability_state import QueuedMcpProbe
from .runner import BenchmarkRunner
from .suites import benchmark_cases


class OfflineProvider:
    """Stable report identity; each scenario supplies its own local transport."""


class ScenarioSession:
    def __init__(self, session, scenario):
        self.session, self.scenario = session, scenario

    def run_turn(self, prompt, *, stop_requested):
        try:
            result = self.session.run_turn(
                prompt,
                stop_requested=lambda: stop_requested() or self.scenario.stop.is_set(),
            )
            self.scenario.verify()
            if isinstance(self.scenario.tools, QueuedMcpProbe):
                self.scenario.errors.extend(self.scenario.tools.verify())
            return result
        finally:
            self.scenario.close()


class ReliabilityRunner(BenchmarkRunner):
    def __init__(self, *, root, **kwargs):
        super().__init__(
            root=root, client=OfflineProvider(), model="offline-contract-v1", **kwargs
        )
        self.active_case = None
        self.scenario = None

    def run_case(self, case, *, trial_index=1):
        self.active_case = case
        original = self.client, self.run_limits, self.max_steps
        try:
            return super().run_case(case, trial_index=trial_index)
        finally:
            self.client, self.run_limits, self.max_steps = original
            if self.scenario is not None:
                self.scenario.close()

    def _build_tools(self, *, workspace, mode, setup, approvals, allowed_tools):
        self.scenario = make_scenario(self.active_case.task.id, workspace)
        setup.metadata["scenario"] = self.scenario
        self.client = self.scenario.client
        self.run_limits = self.scenario.limits
        self.max_steps = 10

        def confirm(request):
            approved = self.scenario.name != "approval_denial"
            approvals.append(
                ApprovalRecord(
                    request.tool_name,
                    request.risk_level.value,
                    request.summary,
                    request.details,
                    approved,
                )
            )
            if self.scenario.name == "cancel_approval":
                self.scenario.stop.set()
            elif self.scenario.name == "cancel_tool":
                self.scenario.stop_running_tool()
            return approved

        self.scenario.tools = (
            QueuedMcpProbe()
            if self.scenario.name == "mcp_queued_cancellation"
            else ToolRegistry(
                Workspace(workspace),
                confirm_tool=confirm,
            )
        )
        return self.scenario.tools

    def _build_session(self, **kwargs):
        return ScenarioSession(super()._build_session(**kwargs), self.scenario)


def run_offline(args) -> int:
    from pathlib import Path

    from .errors import BenchmarkError

    cases = benchmark_cases("reliability")
    if args.case:
        unknown = set(args.case) - {case.id for case in cases}
        if unknown:
            raise BenchmarkError(
                "unknown benchmark case(s): " + ", ".join(sorted(unknown))
            )
        cases = [case for case in cases if case.id in args.case]
    runner = ReliabilityRunner(
        root=Path(args.workspace),
        runtime=args.runtime,
        repeat_count=args.repeat,
        shuffle_seed=args.shuffle_seed,
        resume_path=Path(args.resume) if args.resume else None,
        require_os_sandbox=args.require_os_sandbox,
    )
    results, path, report = runner.run(cases)
    for result in results:
        print(
            f"{'PASS' if result.passed else 'FAIL'} {result.case_id} trial={result.trial_index}"
        )
        for check in [*result.check_results, *result.tool_correctness_results]:
            for message in check.messages if not check.passed else []:
                print(f"  {check.name}: {message}")
        if not result.passed:
            for reason in result.failure_reasons:
                print(f"  {reason.category}: {reason.message}")
    print(f"Benchmark results: {path}\nBenchmark report: {report}")
    return (
        0
        if len(results) == len(cases) * args.repeat and all(r.passed for r in results)
        else 1
    )
