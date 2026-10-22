from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .evidence import export_evidence
from .gate_policy import load_policy, strict_json
from .gate_validation import validate_reports
from .gates import evaluate_gate
from .persistence import atomic_write_text


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mikucli.evaluation.bench gate")
    parser.add_argument(
        "--policy", required=True, help="offline, canary, release, or policy JSON path"
    )
    parser.add_argument(
        "--report",
        action="append",
        default=[],
        help="Benchmark JSON report; repeat for disjoint suites",
    )
    parser.add_argument("--baseline", help="Reviewed sanitized evidence artifact")
    parser.add_argument(
        "--output",
        help="Write only sanitized aggregate/trial evidence and gate decision",
    )
    parser.add_argument(
        "--check-policy",
        action="store_true",
        help="Validate catalog IDs and policy without running evals",
    )
    args = parser.parse_args(argv)
    try:
        policy = load_policy(args.policy)
        if args.check_policy:
            print(f"Valid policy {policy['name']}: {len(policy['cases'])} cases")
            return 0
        if not args.report:
            parser.error("at least one --report is required")
        reports = [strict_json(Path(path)) for path in args.report]
        validate_reports(reports)
        evidence = export_evidence(reports)
        baseline = strict_json(Path(args.baseline)) if args.baseline else None
        decision = evaluate_gate(policy, evidence, baseline)
        evidence["gate"] = decision
        if args.output:
            atomic_write_text(
                Path(args.output),
                json.dumps(evidence, indent=2, allow_nan=False) + "\n",
            )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"mikucli eval gate: invalid evidence or policy: {exc}", file=sys.stderr)
        return 2
    print(f"{'PASS' if decision['passed'] else 'FAIL'} policy={policy['name']}")
    for failure in decision["failures"]:
        print(f"  {failure}")
    return 0 if decision["passed"] else 1
