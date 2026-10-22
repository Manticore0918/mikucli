# ADR 0006: Protected task state and checked compression

Status: Accepted

## Context

Free-form summaries can omit constraints or exact identifiers, and relevance retrieval can exclude an
otherwise correct summary. Previously, empty map responses were discarded and durable facts were extracted
from an already compressed summary. Successful compression cleared the original in-process entries.

## Decision

Assign each MemoryEntry a stable ID and each SessionMemory instance a session scope. Maintain a separate
Protected Task State ledger with stable record IDs, exact source quotes, source roles, provenance, lifecycle
status, and status-change evidence. Capture user requests verbatim at ingestion; recognize explicit constraint
and decision lines, backtick identifiers and hashes, and failed-tool blockers. Preserve full requests as active
task records until explicitly completed. Pattern recognition is deliberately conservative, not a semantic
guarantee. Model-extracted records are unconfirmed interpretations and cannot overwrite known records.

Include active ledger records directly in each prompt. Tool and assistant evidence does not acquire user
instruction authority. Completing or superseding records requires an exact source quote. Retiring constraints,
decisions, or identifiers requires a subsequent user source. Evidence proves provenance, not the semantic
correctness of a lifecycle decision. Keep retired records and their evidence in archived snapshots. Subagent
history clearing starts a fresh scope so evidence cannot leak into another assigned step.

Use `.mikucli/session_memory.sqlite3` as a separate session source archive, leaving the existing fact-store
schema and its migration/recovery contract intact. Journal the exact input batch and current ledger before
LLM calls. These input checkpoints survive crashes without pretending that a summary has been committed.
Archive canonical JSON including native tool-call arguments, tool results, metadata, timestamps, and IDs,
protected by SHA-256 checksums. Source IDs are immutable within a scope. A new session never automatically
loads another session's archive.

Compression prepares a proposal against a source-checksum manifest and a ledger snapshot. Split large
entries into identified parts and require each JSON map response to cover every supplied part exactly once.
Validate nonempty summaries, strict structure, source references, and exact extraction quotes. Allow one
repair of malformed map output; exhausted validation or provider failures abort without eviction. Merge
large summary sets in bounded groups; failure to make progress or an empty merge also aborts. Source
coverage is a mechanical check, not proof that the narrative contains every important fact.

Validate the final proposal against the current source manifest and every existing protected record.
Atomically commit exact sources, a validated summary, and the complete ledger snapshot in one SQLite
transaction with full synchronization. Promote validated workspace-scoped durable quotes directly from
original user evidence to the existing transactional Long-Term Memory store. Fact promotion is a separate
transaction: failure retains original in-process entries, and an already committed archive remains usable.
Only after these steps may the session publish the new ledger/summary and clear compression source entries.
Lifecycle updates likewise commit their source evidence and ledger snapshot before becoming visible.

Offer session-local `recall_session_source` and `update_task_state` capabilities alongside either built-in or
MCP tools. Source lookup is scoped, checksummed and paginated; it preserves the underlying tool registry's
approval and team serialization behavior. Offline `python -m mikucli.memory sessions|source` provides
recovery without credentials. Run logs expose the scope ID. Archives retain raw task/tool data on disk;
there is no automatic archive expiry. Fact-store backup/restore does not include the separate archive file.

Before every model call, estimate the complete serialized prompt and tool schemas conservatively using
UTF-8 bytes plus framing overhead. Reserve one eighth of the context window, at most 4096 tokens, and
apply the same response cap in ModelClient. Both runtimes attempt checked compression before sending an
oversized prompt. Optional retrieved records are admitted whole only while they fit. If required state,
recent rounds, or tool schemas cannot fit, return `limit_reached/context_window_exceeded` without silently
discarding protected records. Estimates may stop a request earlier than a provider-specific tokenizer would;
they are not exact token accounting. Compression and repair calls share the existing task budget.

## Verification

Regression tests deliberately omit constraints from narrative summaries and return no retrieval matches;
protected records remain present. Tests reject missing/duplicate source parts, empty maps, altered identifiers,
unknown references, and altered manifests/ledger records. They cover original-evidence fact extraction,
tool/user provenance, lifecycle evidence, oversized tool arguments, paginated exact lookup, scope isolation,
archive rollback, source tampering, interrupted SQLite transactions, and offline recovery. Both native and
LangGraph runtimes exercise preflight compression and refusal of oversized protected state.
