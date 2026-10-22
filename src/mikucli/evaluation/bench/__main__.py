from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mikucli.codebase.embeddings import OllamaEmbeddingClient
from mikucli.config import ConfigError, load_config
from mikucli.llm import ModelClient
from mikucli.memory import MemoryRetriever

from .models import EvalPrice
from .runner import BenchmarkError, run_benchmarks
from .suites import benchmark_cases


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m mikucli.evaluation.bench")
    parser.add_argument("--workspace", default=".", help="Workspace that receives benchmark result files.")
    parser.add_argument("--model", default=None, help="Model name. Defaults to mikucli config.")
    parser.add_argument(
        "--api-key",
        default=None,
        help="Model provider API key. Prefer MIKUCLI_API_KEY to avoid shell history exposure.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Full OpenAI-compatible Chat Completions URL. Defaults to MIKUCLI_BASE_URL.",
    )
    parser.add_argument("--env-file", default=None, help="Additional .env file with high priority.")
    parser.add_argument("--max-steps", type=int, default=30, help="Maximum ReAct tool loop steps per agent turn.")
    parser.add_argument(
        "--runtime",
        choices=("native", "langgraph"),
        default="native",
        help="Single-agent runtime used by selected benchmark cases.",
    )
    parser.add_argument(
        "--suite",
        choices=("core", "stress"),
        default="core",
        help="Benchmark task suite to run.",
    )
    parser.add_argument("--case", action="append", default=None, help="Benchmark case id to run. May be repeated.")
    parser.add_argument("--repeat", type=int, default=1, help="Run each selected case this many times.")
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help="Shuffle case trials deterministically with this seed.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume a stopped or interrupted benchmark JSON report.",
    )
    parser.add_argument(
        "--require-os-sandbox",
        action="store_true",
        help="Refuse to run unless an OS-enforced sandbox backend is available.",
    )
    parser.add_argument(
        "--prompt-token-price-per-million",
        type=float,
        default=None,
        help="Money price per one million prompt tokens. Enables estimated spend reporting.",
    )
    parser.add_argument(
        "--completion-token-price-per-million",
        type=float,
        default=None,
        help="Money price per one million completion tokens. Enables estimated spend reporting.",
    )
    parser.add_argument("--list", action="store_true", help="List benchmark cases and exit.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list:
        for case in benchmark_cases(args.suite):
            print(case.id)
        return 0

    workspace = Path(args.workspace)
    try:
        config = load_config(
            workspace,
            args.model,
            Path(args.env_file) if args.env_file else None,
            None,
            api_key=args.api_key,
            base_url=args.base_url,
            runtime_overrides=vars(args),
        )
    except ConfigError as exc:
        print(f"mikucli bench: {exc.english}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"mikucli bench: {exc}", file=sys.stderr)
        return 2

    client = ModelClient(api_key=config.api_key, base_url=config.base_url, request_policy=config.request_policy)
    memory_retriever = MemoryRetriever(
        embedding_client=OllamaEmbeddingClient(
            model=config.embedding_model,
            base_url=config.ollama_base_url,
            request_policy=config.request_policy,
        ),
        embedding_model=config.embedding_model,
    )
    try:
        price = EvalPrice(
            prompt_token_price_per_million=args.prompt_token_price_per_million,
            completion_token_price_per_million=args.completion_token_price_per_million,
        )
    except ValueError as exc:
        print(f"mikucli bench: {exc}", file=sys.stderr)
        return 2
    if price.prompt_token_price_per_million is None and price.completion_token_price_per_million is None:
        price = None
    try:
        results, result_path, report_path = run_benchmarks(
            root=config.workspace,
            client=client,
            model=config.model,
            case_ids=set(args.case) if args.case else None,
            max_steps=args.max_steps,
            context_window_tokens=config.context_window_tokens,
            memory_retriever=memory_retriever,
            price=price,
            repeat_count=args.repeat,
            shuffle_seed=args.shuffle_seed,
            resume_path=Path(args.resume) if args.resume else None,
            require_os_sandbox=args.require_os_sandbox,
            runtime=args.runtime,
            suite=args.suite,
            run_limits=config.run_limits,
        )
    except BenchmarkError as exc:
        print(f"mikucli bench: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"mikucli bench: benchmark run failed: {exc}", file=sys.stderr)
        return 2

    passed = sum(1 for result in results if result.passed)
    total = len(results)
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        trial = f" trial={result.trial_index}/{result.trial_count}" if result.trial_count > 1 else ""
        print(f"{status} {result.case_id}{trial} (latency={result.metrics.elapsed_seconds:.3f}s)")
        for check in result.check_results:
            if not check.passed:
                print(f"  - {check.name}: {'; '.join(check.messages)}")
        for reason in result.failure_reasons:
            if reason.category != "check_failed":
                print(f"  - {reason.category} [{reason.source}]: {reason.message}")
    summary = _summarize_cli(results)
    print(f"Benchmark results: {result_path}")
    print(f"Benchmark report: {report_path}")
    print(f"{passed}/{total} benchmark cases passed")
    print(
        "Summary: "
        f"success_rate={summary['success_rate']:.1f}%, "
        f"tool_calls={summary['tool_calls']}, "
        f"model_retries={summary['model_retries']}, "
        f"step_retries={summary['step_retries']}, "
        f"latency={summary['latency']:.3f}s, "
        f"tokens={summary['tokens']}, "
        f"estimated_spend={summary['estimated_spend']}"
    )
    return 0 if passed == total else 1


def _summarize_cli(results) -> dict[str, object]:
    total = len(results)
    passed = sum(1 for result in results if result.passed)
    prompt_tokens = _sum_optional(result.metrics.cost.prompt_tokens for result in results)
    completion_tokens = _sum_optional(result.metrics.cost.completion_tokens for result in results)
    total_tokens = _sum_optional(result.metrics.cost.total_tokens for result in results)
    estimated_spend = _sum_optional(
        result.metrics.estimated_spend.total if result.metrics.estimated_spend is not None else None
        for result in results
    )
    return {
        "success_rate": (passed / total * 100) if total else 0.0,
        "tool_calls": sum(result.metrics.tool_call_count for result in results),
        "model_retries": sum(result.metrics.model_retries for result in results),
        "step_retries": sum(result.metrics.step_retries for result in results),
        "latency": sum(result.metrics.elapsed_seconds for result in results),
        "tokens": f"prompt={_fmt_optional(prompt_tokens)}, completion={_fmt_optional(completion_tokens)}, total={_fmt_optional(total_tokens)}",
        "estimated_spend": _fmt_optional_float(estimated_spend),
    }


def _sum_optional(values) -> int | float | None:
    collected = list(values)
    if not collected or any(value is None for value in collected):
        return None
    return sum(value for value in collected if value is not None)


def _fmt_optional(value: object) -> str:
    return "unknown" if value is None else str(value)


def _fmt_optional_float(value: object) -> str:
    if value is None:
        return "unknown"
    return f"{float(value):.8f}".rstrip("0").rstrip(".")


if __name__ == "__main__":
    raise SystemExit(main())
