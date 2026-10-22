from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]


def workflow(name):
    return yaml.load(
        (ROOT / ".github/workflows" / name).read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )


def test_pr_ci_gates_both_runtimes_without_provider_secrets():
    document = workflow("ci.yml")
    job = document["jobs"]["test"]
    assert set(job["strategy"]["matrix"]["os"]) == {"ubuntu-latest", "windows-latest"}
    steps = job["steps"]
    code = next(
        step["run"]
        for step in steps
        if step.get("name") == "Gate repeated offline reliability contracts"
    )
    compile(code, "ci-offline-step", "exec")
    assert '"native", "langgraph"' in code
    assert "repeat_count=3" in code
    assert "gate_cli" in code
    assert "secrets." not in (ROOT / ".github/workflows/ci.yml").read_text(
        encoding="utf-8"
    )
    upload = next(
        step
        for step in steps
        if step.get("uses", "").startswith("actions/upload-artifact")
    )
    assert upload["with"]["path"] == "eval-artifacts/*.json"


def test_live_workflow_only_runs_on_trusted_triggers_and_uploads_sanitized_files():
    document = workflow("eval-live.yml")
    assert set(document["on"]) == {"schedule", "workflow_dispatch"}
    steps = document["jobs"]["evaluate"]["steps"]
    live = next(
        step for step in steps if step.get("name") == "Run bounded live campaign"
    )
    compile(live["run"], "ci-live-step", "exec")
    assert "--max-spend" in live["run"]
    assert "--baseline" in live["run"]
    assert "MIKUCLI_EVAL_API_KEY" in live["env"]["MIKUCLI_API_KEY"]
    upload = next(
        step
        for step in steps
        if step.get("uses", "").startswith("actions/upload-artifact")
    )
    assert upload["with"]["path"] == "eval-artifacts/*.json"
    assert all("env" not in step for step in steps if step is not live)
