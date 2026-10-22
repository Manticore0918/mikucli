"""Bounded live campaigns over the case selection in a versioned policy."""

from __future__ import annotations

import copy
from dataclasses import asdict, replace
from pathlib import Path
from threading import Event

from mikucli.llm import ModelClient

from .evidence import export_evidence, number
from .gate_policy import catalog, strict_json, validate_policy
from .gate_validation import validate_evidence, validate_reports
from .gates import evaluate_gate
from .metadata import build_run_metadata
from .models import EvalPrice
from .runner import BenchmarkRunner


def run_campaign(
    *,
    config,
    policy,
    repeat_count,
    shuffle_seed,
    runtime,
    maximum_spend,
    baseline=None,
    provider="configured",
    model_revision=None,
    bootstrap_baseline=False,
):
    validate_policy(policy)
    if policy["require_baseline"] and baseline is None and not bootstrap_baseline:
        raise ValueError(
            "release campaign requires a baseline or explicit bootstrap before spending"
        )
    limits = replace(config.run_limits, max_spend=maximum_spend)
    price = EvalPrice(
        limits.prompt_price_per_million, limits.completion_price_per_million
    )
    if (
        price.prompt_token_price_per_million is None
        or price.completion_token_price_per_million is None
    ):
        raise ValueError(
            "live campaigns require both token prices for spend accounting"
        )
    if type(repeat_count) is not int or repeat_count < policy["minimum_repeat_count"]:
        raise ValueError("campaign repeat count is below policy")
    if baseline is not None:
        validate_evidence(baseline)
        if not evaluate_gate(policy, copy.deepcopy(baseline), copy.deepcopy(baseline))[
            "passed"
        ]:
            raise ValueError("baseline does not meet the selected policy")
    client = ModelClient(
        config.api_key, config.base_url, request_policy=config.request_policy
    )
    client.evaluation_provider = provider
    client.evaluation_model_revision = model_revision
    client.campaign_spend_limit = maximum_spend
    known = catalog()
    grouped = {}
    for case_id in sorted(policy["cases"]):
        case = known[case_id]
        if case.task.suite == "reliability":
            raise ValueError(
                "live campaigns cannot substitute for the separate offline contract gate"
            )
        grouped.setdefault(case.task.suite, []).append(case)
    runners = []
    for suite, cases in sorted(grouped.items()):
        runner = BenchmarkRunner(
            root=config.workspace,
            client=client,
            model=config.model,
            max_steps=config.max_steps if hasattr(config, "max_steps") else 30,
            context_window_tokens=config.context_window_tokens,
            price=price,
            runtime=runtime,
            repeat_count=repeat_count,
            shuffle_seed=shuffle_seed,
            run_limits=limits,
        )
        planned = build_run_metadata(
            root=config.workspace,
            model=config.model,
            max_steps=runner.max_steps,
            context_window_tokens=config.context_window_tokens,
            cases=cases,
            repeat_count=repeat_count,
            shuffle_seed=shuffle_seed,
            client=client,
            memory_retriever=None,
            price=asdict(price),
            runtime=runtime,
            run_limits=asdict(limits),
        )
        if baseline is not None:
            matching = [
                report for report in baseline["reports"] if report["suite"] == suite
            ]
            if len(matching) != 1 or any(
                matching[0][field] != expected
                for field, expected in (
                    ("configuration_fingerprint", planned["configuration_fingerprint"]),
                    ("suite_fingerprint", planned["suite"]["fingerprint"]),
                    ("python", planned["environment"]["python"]),
                    ("platform", planned["environment"]["platform"].split("-")[0]),
                )
            ):
                raise ValueError(
                    "baseline configuration/harness/environment is incompatible; no provider calls made"
                )
        runners.append((runner, cases))
    spent = 0.0
    uncertain = False
    stop = Event()
    paths = []
    interrupted = False

    def before_case(_):
        remaining = maximum_spend - spent
        if remaining <= 0 or uncertain:
            stop.set()
            return
        runner.run_limits = replace(limits, max_spend=remaining)

    def after_case(result):
        nonlocal spent, uncertain
        amount = (
            number(result.metrics.estimated_spend.total)
            if result.metrics.estimated_spend
            else None
        )
        if amount is None:
            uncertain = True
        else:
            spent += amount
        if uncertain or spent >= maximum_spend:
            stop.set()

    for runner, cases in runners:
        if stop.is_set():
            break
        runner.stop_requested = stop.is_set
        runner.on_case_started = before_case
        runner.on_case_finished = after_case
        try:
            runner.run(cases)
        except KeyboardInterrupt:
            interrupted = True
            stop.set()
        finally:
            path = runner.results_root / f"{runner.run_id}.json"
            if path.exists():
                paths.append(path)
        if interrupted:
            break
    reports = [strict_json(Path(path)) for path in paths]
    validate_reports(reports)
    evidence = export_evidence(reports)
    # An interrupted first trial can leave zero case results; preserve the report
    # and fail explicitly instead of accidentally accepting an empty evaluation.
    if not reports or any(not report["results"] for report in reports):
        decision = {
            "passed": False,
            "policy": policy["name"],
            "failures": ["campaign produced incomplete trial evidence"],
            "comparisons": {},
        }
    else:
        decision = evaluate_gate(policy, evidence, baseline)
    evidence["gate"] = decision
    evidence["campaign"] = {
        "reported_spend": spent,
        "spend_unknown": uncertain,
        "spend_limit": maximum_spend,
        "interrupted": interrupted,
    }
    return evidence
