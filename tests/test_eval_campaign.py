from __future__ import annotations

import copy
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest

from mikucli.evaluation.bench.campaign import run_campaign
from mikucli.evaluation.bench.campaign_cli import main
from mikucli.http_requests import RequestPolicy
from mikucli.llm import ModelClient
from mikucli.run_control import RunLimits

CASE = "repo_inspection:built_in_single_agent"


def config(root, *, prices=True):
    return SimpleNamespace(
        workspace=root,
        model="offline-provider-model",
        api_key="fixture",
        base_url="https://offline.invalid/chat",
        context_window_tokens=128000,
        request_policy=RequestPolicy(max_retries=0),
        run_limits=RunLimits(
            max_duration_seconds=10,
            prompt_price_per_million=1 if prices else None,
            completion_price_per_million=1 if prices else None,
        ),
    )


def policy():
    return {
        "schema_version": 1,
        "name": "fixture-campaign",
        "minimum_repeat_count": 3,
        "require_baseline": False,
        "require_usage": True,
        "require_clean_revision": False,
        "maximum_pass_rate_drop": 0,
        "cases": {CASE: {"minimum_pass_rate": 1}},
    }


def transport_client(requests, usage=True):
    def handler(request):
        requests.append(request)
        payload = {
            "choices": [
                {
                    "message": {
                        "content": "Acorn Ledger uses src/acorn and tests/test_calculator.py."
                    }
                }
            ]
        }
        if usage:
            payload["usage"] = {
                "prompt_tokens": 2,
                "completion_tokens": 1,
                "total_tokens": 3,
            }
        return httpx.Response(200, json=payload)

    return lambda *args, **kwargs: ModelClient(
        *args, **kwargs, transport=httpx.MockTransport(handler)
    )


def run(root, **overrides):
    arguments = dict(
        config=config(root),
        policy=policy(),
        repeat_count=3,
        shuffle_seed=1729,
        runtime="native",
        maximum_spend=0.1,
    )
    arguments.update(overrides)
    return run_campaign(**arguments)


def test_campaign_runs_repeats_and_exports_sanitized_evidence(tmp_path):
    requests = []
    with patch(
        "mikucli.evaluation.bench.campaign.ModelClient",
        side_effect=transport_client(requests),
    ):
        evidence = run(tmp_path)
    assert evidence["gate"]["passed"]
    assert len(requests) == 3
    assert evidence["campaign"]["reported_spend"] == pytest.approx(0.000009)
    assert len(evidence["reports"][0]["cases"][CASE]["trial_results"]) == 3


def test_campaign_budget_exhaustion_keeps_partial_trials_and_stops_requests(tmp_path):
    requests = []
    with patch(
        "mikucli.evaluation.bench.campaign.ModelClient",
        side_effect=transport_client(requests),
    ):
        evidence = run(tmp_path, maximum_spend=0.000001)
    assert not evidence["gate"]["passed"]
    assert len(requests) == 1
    assert evidence["reports"][0]["status"] == "stopped"
    assert evidence["reports"][0]["cases"][CASE]["trials"] == 1


def test_unknown_usage_stops_campaign_without_scheduling_more_work(tmp_path):
    requests = []
    with patch(
        "mikucli.evaluation.bench.campaign.ModelClient",
        side_effect=transport_client(requests, usage=False),
    ):
        evidence = run(tmp_path)
    assert not evidence["gate"]["passed"]
    assert evidence["campaign"]["spend_unknown"]
    assert len(requests) == 1


def test_prices_and_repeat_validation_happen_before_any_provider_call(tmp_path):
    requests = []
    with patch(
        "mikucli.evaluation.bench.campaign.ModelClient",
        side_effect=transport_client(requests),
    ):
        with pytest.raises(ValueError, match="prices"):
            run(tmp_path, config=config(tmp_path, prices=False))
        with pytest.raises(ValueError, match="repeat count"):
            run(tmp_path, repeat_count=1)
    assert requests == []


def test_incompatible_baseline_is_rejected_before_spending(tmp_path):
    requests = []
    with patch(
        "mikucli.evaluation.bench.campaign.ModelClient",
        side_effect=transport_client(requests),
    ):
        baseline = run(tmp_path)
        requests.clear()
        with pytest.raises(ValueError, match="incompatible"):
            run(tmp_path, maximum_spend=0.2, baseline=copy.deepcopy(baseline))
    assert requests == []


def test_release_api_requires_explicit_bootstrap_before_spending(tmp_path):
    release_policy = policy()
    release_policy["require_baseline"] = True
    requests = []
    with patch(
        "mikucli.evaluation.bench.campaign.ModelClient",
        side_effect=transport_client(requests),
    ):
        with pytest.raises(ValueError, match="explicit bootstrap"):
            run(tmp_path, policy=release_policy)
        evidence = run(tmp_path, policy=release_policy, bootstrap_baseline=True)
    assert len(requests) == 3
    assert not evidence["gate"]["passed"]
    assert any("baseline" in failure for failure in evidence["gate"]["failures"])


def test_release_cli_requires_baseline_before_loading_credentials(tmp_path):
    with patch(
        "mikucli.evaluation.bench.campaign_cli.load_config",
        side_effect=AssertionError("configuration loaded"),
    ):
        assert (
            main(
                [
                    "--profile",
                    "release",
                    "--max-spend",
                    "1",
                    "--output",
                    str(tmp_path / "evidence.json"),
                ]
            )
            == 2
        )
