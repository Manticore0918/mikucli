from __future__ import annotations

from typing import Literal

from .catalog import expand_benchmark_cases
from .errors import BenchmarkError
from .models import BenchmarkCase, BenchmarkTask
from .stress_tasks import stress_benchmark_tasks
from .tasks import all_benchmark_tasks


BenchmarkSuite = Literal["core", "stress"]


def benchmark_tasks(suite: BenchmarkSuite = "core") -> list[BenchmarkTask]:
    if suite == "core":
        return all_benchmark_tasks()
    if suite == "stress":
        return stress_benchmark_tasks()
    raise BenchmarkError(f"unknown benchmark suite: {suite}")


def benchmark_cases(suite: BenchmarkSuite = "core") -> list[BenchmarkCase]:
    return expand_benchmark_cases(benchmark_tasks(suite))
