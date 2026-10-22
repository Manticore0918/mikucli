from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mikucli.config import ConfigError, load_config

from .campaign import run_campaign
from .gate_policy import load_policy, strict_json
from .persistence import atomic_write_text


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mikucli.evaluation.bench campaign")
    parser.add_argument("--profile", choices=("canary", "release"), default="canary")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--runtime", choices=("native", "langgraph"), default="native")
    parser.add_argument("--model")
    parser.add_argument(
        "--provider", default="configured", help="Stable provider/deployment label"
    )
    parser.add_argument(
        "--model-revision",
        help="Explicit deployment revision, when available; aliases may drift",
    )
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--shuffle-seed", type=int, default=1729)
    parser.add_argument(
        "--max-spend",
        type=float,
        required=True,
        help="Campaign-wide observed spend limit; not a billing ceiling",
    )
    parser.add_argument("--max-duration", type=float, default=180)
    parser.add_argument("--max-tokens", type=int, default=30000)
    parser.add_argument("--max-output-bytes", type=int, default=65536)
    parser.add_argument("--request-timeout", type=float, default=30)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-max-delay", type=float, default=2)
    parser.add_argument("--prompt-token-price-per-million", type=float)
    parser.add_argument("--completion-token-price-per-million", type=float)
    parser.add_argument(
        "--baseline",
        help="Reviewed sanitized evidence from the same profile/configuration",
    )
    parser.add_argument(
        "--bootstrap-baseline",
        action="store_true",
        help="Collect initial release evidence; gate still fails without a baseline",
    )
    parser.add_argument(
        "--output", required=True, help="Sanitized CI evidence output path"
    )
    args = parser.parse_args(argv)
    try:
        policy = load_policy(args.profile)
        if (
            policy["require_baseline"]
            and not args.baseline
            and not args.bootstrap_baseline
        ):
            raise ValueError(
                "release profile requires --baseline or explicit --bootstrap-baseline before spending"
            )
        config = load_config(
            Path(args.workspace), args.model, None, None, runtime_overrides=vars(args)
        )
        evidence = run_campaign(
            config=config,
            policy=policy,
            repeat_count=args.repeat
            if args.repeat is not None
            else policy["minimum_repeat_count"],
            shuffle_seed=args.shuffle_seed,
            runtime=args.runtime,
            maximum_spend=args.max_spend,
            baseline=strict_json(Path(args.baseline)) if args.baseline else None,
            provider=args.provider,
            model_revision=args.model_revision,
            bootstrap_baseline=args.bootstrap_baseline,
        )
        atomic_write_text(
            Path(args.output), json.dumps(evidence, indent=2, allow_nan=False) + "\n"
        )
    except (ConfigError, ValueError, OSError, KeyError, TypeError) as exc:
        print(f"mikucli eval campaign: {exc}", file=sys.stderr)
        return 2
    decision = evidence["gate"]
    print(f"{'PASS' if decision['passed'] else 'FAIL'} profile={args.profile}")
    for failure in decision["failures"]:
        print(f"  {failure}")
    print(f"Sanitized evidence: {args.output}")
    return 0 if decision["passed"] else 1
