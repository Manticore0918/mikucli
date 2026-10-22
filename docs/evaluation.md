# Evaluation and release evidence

MikuCLI separates runner contracts from live-model task quality. A successful
runtime outcome is necessary for ordinary coding tasks; it is not proof of a
correct patch. An intentional denial, cancellation, or limit test instead declares
its expected status and reason, then checks side effects and cleanup independently.

## Suites

| Suite | Tasks | Cases per runtime selection | Purpose |
| --- | ---: | ---: | --- |
| `core` | 6 | 12 | Basic inspection, fixes, editing, retrieval, and MCP |
| `stress` | 6 | 12 | Multi-file work, distractors, permission recovery, refactoring, parallel work, retrieval ambiguity |
| `realworld` | 7 | 14 | Pinned repository defects, behavior preservation, bad-patch review, repository prompt injection |
| `reliability` | 21 | 21 | Deterministic failure/recovery workflows without credentials or network services |

The runtime switch selects Native or LangGraph for single-agent cases. Multi-agent
cases use the native orchestrator, including when the single-agent runtime setting
is `langgraph`. Case counts distinguish task definitions from task/mode pairs.

The repository pack is the MikuCLI memory subsystem at commit
`564cb7d4b582c025add455cda9c704b9340a1ad2`, licensed Apache-2.0. Its source modules
and checksums are bundled in `src/mikucli/evaluation/bench/fixtures/memory_v1.json`.
Setup injects narrow defects into that snapshot; it never copies the current
working tree or downloads a moving upstream branch. This is a subsystem fixture,
not evidence of performance on arbitrary large repositories or other languages.

Hidden checks stay outside the editable workspace. Only function inputs enter the
generated-code process; expected outputs, primitive decoding, and pass decisions
stay in the parent. Public tests, file-scope checks, structured review verdicts,
and permission invariants provide separate evidence. Scorer tests include broken
solutions that pass public examples, test-file tampering, hardcoded answers,
incorrect review verdicts, and prompt-injection side effects. No LLM judge decides
the safety gate.

## Offline contracts and gates

Run on a clean, committed checkout:

```bash
python -m mikucli.evaluation.bench --suite reliability --repeat 3 --shuffle-seed 1729
python -m mikucli.evaluation.bench --suite reliability --runtime langgraph --repeat 3 --shuffle-seed 1729
python -m mikucli.evaluation.bench gate --policy offline --report PATH_FROM_RUN.json --output .mikucli/evidence/offline.json
```

The contracts cover transient HTTP errors, bounded exhaustion, invalid responses,
request timeouts, cancellation during requests/backoff/approvals/tools, duration,
token/spend/output limits, missing usage, failed workers/reviewers, shared team
budgets, durable restart/restore, concurrent writers, interrupted transactions,
and queued MCP cancellation. Each trial creates fresh transport state and a fresh
workspace. The MCP queue contract exercises the real submission/guard mechanism
locally; the `core` MCP tasks exercise an actual fixture server.

Deadline fault probes advance the active budget's elapsed clock after the HTTP
request or MCP submission begins, then let the real deadline check cancel it.
This keeps startup/trace-store scheduling from preventing the intended in-flight
condition. It does not relax the production task duration limit. The initial
repeated run that exposed this scenario flaw is retained in `evaluation-history.json`.

An expected task cancellation does not stop the suite. An external suite stop
leaves unfinished work pending. Compatible resume retains completed trials,
including failures; it cannot turn a failed completed trial into a fresh attempt.

```bash
python -m mikucli.evaluation.bench gate --policy offline --check-policy
python -m mikucli.evaluation.bench gate --policy canary --check-policy
python -m mikucli.evaluation.bench gate --policy release --check-policy
```

Policies are committed package resources in `fixtures/policy-*.json`. Policy and
report parsing rejects duplicate JSON keys, non-finite numbers, misspelled fields,
unknown catalog IDs, duplicate trials, contradictory outcomes, and stale scorer
fingerprints. The earlier local `.github/eval-canary-baseline.json` draft is not
used by these commands or CI.

| Policy | Minimum repeats per case | Required cases | Quality | Baseline |
| --- | ---: | ---: | --- | --- |
| `offline` | 3 | 21 | All contracts pass | Optional |
| `canary` | 3 | 9 | All selected tasks pass | Optional |
| `release` | 10 | 38 | At least 95% per functional case; injection cases 100% | Required |

Every policy requires complete evidence from an identified, clean Git revision.
Observed failures of safety, scope, tool correctness, runtime outcomes, or runner
contracts always fail the gate, regardless of average pass rate. At ten trials,
the 95% functional threshold requires ten passes; twenty trials permit one ordinary
functional failure. This does not relax critical invariants.

Live policies require complete reported usage and estimated spend. Unknown usage
is not zero. With a compatible baseline, the policies reject any pass-rate drop,
median latency/token/spend increases above 25%, and p95 latency increases above 50%.
These are initial engineering thresholds, not statistical significance claims.
Custom policies can set absolute per-case ceilings and `minimum_ci95_low`.

## Bounded live campaigns

Use normal provider configuration (`MIKUCLI_API_KEY`, `MIKUCLI_MODEL`, and the full
Chat Completions `MIKUCLI_BASE_URL`) plus both token prices. Prices may be passed as
CLI flags or the existing `MIKUCLI_PROMPT_TOKEN_PRICE_PER_MILLION` and
`MIKUCLI_COMPLETION_TOKEN_PRICE_PER_MILLION` environment variables.

```bash
python -m mikucli.evaluation.bench campaign --profile canary --provider your-deployment --runtime native --max-spend 1 --output .mikucli/evidence/canary-native.json
python -m mikucli.evaluation.bench campaign --profile release --provider your-deployment --runtime native --max-spend 5 --baseline REVIEWED_NATIVE.json --output .mikucli/evidence/release-native.json
```

The profile selects disjoint cases across suites, reuses a fixed trial schedule,
and sets shared per-task duration, token, output, and spend limits. The campaign
tracks total reported spend across all cases and lowers each next task's remaining
allowance. Missing usage or an exhausted budget stops further scheduling and
produces failing partial evidence. The final billed request and unreported retry
or canceled-request charges can exceed the observed budget; it is not a provider
billing ceiling. Each CI runtime job has its own configured campaign allowance.

Campaigns use deterministic fixture retrieval where the selected task needs it;
they do not require a separate Ollama server. A custom deployment's embedding
integration needs its own coverage before being claimed supported.

An initial release baseline must be collected explicitly:

```bash
python -m mikucli.evaluation.bench campaign --profile release --provider your-deployment --runtime native --max-spend 5 --bootstrap-baseline --output .mikucli/evidence/bootstrap-native.json
```

This still exits with a failed gate because no baseline was supplied. Review the
complete per-case evidence before retaining it as the reference. Never promote
an incomplete or failing-quality run. Subsequent release runs reject missing or
incompatible references before making provider calls.

Comparisons require the same task/scorer fingerprints, selected case contracts,
provider/model settings, budgets, prices, repeat count, trial schedule, runtime,
OS family, and Python version. Git revisions may differ: comparing a candidate
change against its reference is the purpose of the gate. Supply `--model-revision`
when the provider exposes a pinned deployment revision. An unchanged alias does
not prove unchanged model weights; without a revision that limitation remains.

Three repeats are a smoke check. Increase sampling for unstable or consequential
cases using `--repeat`, with a matching reference schedule. Reports give per-case
Wilson intervals, flakiness, median/p95 latency, usage/spend, HTTP/model attempts,
and failure categories. Intervals across a small handpicked suite do not establish
high reliability on unseen tasks; paired win/loss counts are descriptive.

## CI and artifact handling

PR CI runs unit/scorer tests on Ubuntu and Windows with Python 3.11/3.12, validates
all policies, then gates three repeats of every offline contract for both runtime
selections. These jobs need no provider secrets.

`eval-live.yml` runs only on a schedule or explicit workflow dispatch. The weekly
schedule stays disabled until repository variable `MIKUCLI_LIVE_EVAL_ENABLED` is
`true`. Configure:

- Secret: `MIKUCLI_EVAL_API_KEY`.
- Variables: `MIKUCLI_EVAL_BASE_URL`, `MIKUCLI_EVAL_MODEL`,
  `MIKUCLI_EVAL_PROMPT_PRICE`, `MIKUCLI_EVAL_COMPLETION_PRICE`,
  `MIKUCLI_EVAL_MAX_SPEND`.
- Optional labels: `MIKUCLI_EVAL_PROVIDER`, `MIKUCLI_EVAL_MODEL_REVISION`.
- Release reference directory: `MIKUCLI_EVAL_BASELINE_DIRECTORY`, containing
  reviewed `native.json` and `langgraph.json` artifacts for that configuration.

The scheduled profile is `canary`; manually select `release` for release evidence.
Run each deployment/model configuration intended for the support matrix. One
provider's pass does not certify another provider. Keep these workflows as required
checks in the repository's release process; this change does not alter GitHub branch
protection or publish a release.

Only allowlisted evidence JSON is uploaded: identities/fingerprints, revision and
environment identifiers, numeric metrics, outcome/failure categories, and gate
decisions. Prompts, answers, source contents, tool arguments, endpoints, messages,
diffs, and local file paths are omitted. Raw reports, traces, and workspaces stay
under `.mikucli` and are not uploaded by these workflows. Sanitized artifacts are
reviewable references, not cryptographically authenticated attestations.

## Current evidence and support claims

The repository includes deterministic scorer/runner checks and workflows for both
runtime selections. LangGraph remains experimental. The new live task pack needs
fresh provider runs before any deployment can be described as release-qualified.
Local deterministic evidence is recorded separately from live-provider results.

The generated-code check backend remains the documented Python audit-hook runner;
it is not an OS sandbox. This work does not establish safe arbitrary-code execution
for hostile repositories or certify production readiness by itself.
