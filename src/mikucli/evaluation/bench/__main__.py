from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mikucli.config import ConfigError, load_config
from mikucli.llm import BigModelClient

from .runner import BenchmarkError, run_benchmarks
from .tasks import all_benchmark_cases


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m mikucli.evaluation.bench")
    parser.add_argument("--workspace", default=".", help="Workspace that receives benchmark result files.")
    parser.add_argument("--model", default=None, help="Model name. Defaults to mikucli config.")
    parser.add_argument("--env-file", default=None, help="Additional .env file with high priority.")
    parser.add_argument("--max-steps", type=int, default=30, help="Maximum ReAct tool loop steps per agent turn.")
    parser.add_argument("--case", action="append", default=None, help="Benchmark case id to run. May be repeated.")
    parser.add_argument("--list", action="store_true", help="List benchmark cases and exit.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list:
        for case in all_benchmark_cases():
            print(case.id)
        return 0

    workspace = Path(args.workspace)
    try:
        config = load_config(
            workspace,
            args.model,
            Path(args.env_file) if args.env_file else None,
            None,
        )
    except ConfigError as exc:
        print(f"mikucli bench: {exc.english}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"mikucli bench: {exc}", file=sys.stderr)
        return 2

    client = BigModelClient(api_key=config.api_key, base_url=config.base_url)
    try:
        results, result_path = run_benchmarks(
            root=config.workspace,
            client=client,
            model=config.model,
            case_ids=set(args.case) if args.case else None,
            max_steps=args.max_steps,
            context_window_tokens=config.context_window_tokens,
        )
    except BenchmarkError as exc:
        print(f"mikucli bench: {exc}", file=sys.stderr)
        return 2

    passed = sum(1 for result in results if result.passed)
    total = len(results)
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"{status} {result.case_id} ({result.elapsed_seconds:.3f}s)")
        for check in result.check_results:
            if not check.passed:
                print(f"  - {check.name}: {'; '.join(check.messages)}")
    print(f"Benchmark results: {result_path}")
    print(f"{passed}/{total} benchmark cases passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
