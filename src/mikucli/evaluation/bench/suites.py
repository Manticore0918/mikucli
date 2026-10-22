from __future__ import annotations

from typing import Literal

from .catalog import expand_benchmark_cases
from .errors import BenchmarkError
from .models import BenchmarkCase, BenchmarkTask
from .stress_tasks import stress_benchmark_tasks
from .tasks import all_benchmark_tasks

BenchmarkSuite = Literal["core", "stress", "reliability", "realworld"]


def benchmark_tasks(suite: BenchmarkSuite = "core") -> list[BenchmarkTask]:
    if suite == "core":
        return all_benchmark_tasks()
    if suite == "stress":
        return stress_benchmark_tasks()
    if suite == "reliability":
        from .reliability_tasks import reliability_tasks

        return reliability_tasks()
    if suite == "realworld":
        from .realworld_tasks import realworld_tasks

        return realworld_tasks()
    raise BenchmarkError(f"unknown benchmark suite: {suite}")


def benchmark_cases(suite: BenchmarkSuite = "core") -> list[BenchmarkCase]:
    return expand_benchmark_cases(benchmark_tasks(suite))
