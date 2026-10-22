# ADR 0005: Explicit task outcomes and shared run limits

Status: Accepted

## Context

Step exhaustion returned the same result shape and CLI exit code as normal completion. Team execution
could consume an unsuccessful subagent answer as planner or reviewer content. Provider requests aborted
on the first transient failure, and shell output was accumulated without a size bound.

## Decision

Extend SessionResult compatibly with a status, stable reason, changes, metrics, and error details.
Persist outcomes in run logs and attach them to traces. CLI exit codes distinguish success (0), failure
(1), configuration errors (2), limits (3), and user cancellation (130). One-shot JSON emits one versioned
object to stdout and diagnostics to stderr; approvals fail closed. Positive step counts are required.
Team planners, workers, and reviewers must return a successful runtime result before their output is
consumed. Non-successful runtime results fail benchmarks independently of artifact checks.
Runtime completion is distinct from independently verified answer correctness. Recovered tool failures
remain observable; unresolved approval denial produces failure.

Bind a RunBudget to each top-level turn using context variables. Explicitly propagate that context to
worker threads. Workers, planning, review, and compression share duration, token, spend, and output
accounting. Serialize model calls when usage caps are configured. Stop before further tools or model
calls after a threshold or deadline; retain already completed changes. Usage-based limits account for
provider reports and cannot guarantee a billing ceiling. Missing required usage fails closed.

Use a shared cancellable HTTP path for chat and embeddings: bounded per-attempt deadlines, retry count,
exponential jitter, Retry-After support, and response-size limits. Retry only transient transport errors
and selected transient HTTP statuses. Refuse overly long Retry-After waits instead of retrying early.
Malformed, incomplete, and authentication responses fail explicitly. Embedding failure remains strict.
Dependency preflight checks configuration, storage, and installed Ollama models without a chat request.

Bound shell output while draining both pipes; terminate the entire process tree on cancellation,
timeout, or output exhaustion. Bound other tool output before persistence and display. Approval waits,
queued mutations, and MCP waits observe the same cancellation/deadline, including a guard before an
MCP server dequeues a command. Budget settings and request policy participate in eval fingerprints.

## Verification

Tests exercise transient recovery and exhaustion, Retry-After, cancellation during backoff and requests,
malformed responses, response size, embedding recovery messages, both runtime outcome contracts,
cross-worker and compression usage, process output and duration limits, pending approval release,
JSON framing, denial without stdin, CLI exit codes, and dependency checks without chat calls.
