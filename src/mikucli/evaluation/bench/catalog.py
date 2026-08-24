from __future__ import annotations

import re
from collections.abc import Iterable

from .errors import BenchmarkError
from .models import BenchmarkCase, BenchmarkTask, SessionMode


_TASK_ID = re.compile(r"^[a-z][a-z0-9_]*$")
_METADATA_ID = re.compile(r"^[a-z][a-z0-9_]*$")


def validate_benchmark_tasks(tasks: Iterable[BenchmarkTask]) -> list[BenchmarkTask]:
    """Return a materialized catalog after checking stable IDs and runnable coverage."""
    catalog = list(tasks)
    if not catalog:
        raise BenchmarkError("benchmark task catalog is empty")
    task_ids: set[str] = set()
    for task in catalog:
        if not _TASK_ID.fullmatch(task.id):
            raise BenchmarkError(f"invalid benchmark task id: {task.id!r}")
        if task.id in task_ids:
            raise BenchmarkError(f"duplicate benchmark task id: {task.id}")
        task_ids.add(task.id)
        if not task.title.strip() or not task.prompt.strip():
            raise BenchmarkError(f"benchmark task {task.id} requires a title and prompt")
        if not callable(task.setup):
            raise BenchmarkError(f"benchmark task {task.id} setup is not callable")
        if not task.checks or any(not callable(check) for check in task.checks):
            raise BenchmarkError(f"benchmark task {task.id} requires callable checks")
        if not task.session_modes:
            raise BenchmarkError(f"benchmark task {task.id} has no session modes")
        if any(not isinstance(mode, SessionMode) for mode in task.session_modes):
            raise BenchmarkError(f"benchmark task {task.id} has an invalid session mode")
        if len(task.session_modes) != len(set(task.session_modes)):
            raise BenchmarkError(f"benchmark task {task.id} has duplicate session modes")
        if not _METADATA_ID.fullmatch(task.suite):
            raise BenchmarkError(f"benchmark task {task.id} has an invalid suite")
        if any(not _METADATA_ID.fullmatch(capability) for capability in task.capabilities):
            raise BenchmarkError(f"benchmark task {task.id} has an invalid capability")
    return catalog


def expand_benchmark_cases(tasks: Iterable[BenchmarkTask]) -> list[BenchmarkCase]:
    cases = [
        BenchmarkCase(id=f"{task.id}:{mode.value}", task=task, session_mode=mode)
        for task in validate_benchmark_tasks(tasks)
        for mode in task.session_modes
    ]
    case_ids = [case.id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise BenchmarkError("benchmark catalog produced duplicate case IDs")
    return cases
