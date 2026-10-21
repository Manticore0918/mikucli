# ADR 0004: Transactional long-term memory

Status: Accepted

## Context

The JSON memory store loaded one snapshot per object and rewrote the complete file for each fact.
Independent sessions could lose each other's changes, an interrupted write could truncate the file,
and updating in-memory records before writing could make a failed save appear to be a duplicate on retry.
Worker threads and separate CLI processes both use persistent memory. Compression also promotes facts.

## Decision

Keep the `LongTermMemory.save()` and `records` interface while using the Python standard-library SQLite
driver behind it. Use `.mikucli/long_term_memory.sqlite3`, short connections per operation, full synchronous
writes, and explicit immediate transactions for writes. The SQLite unique constraint on the existing
case-folded, whitespace-normalized key arbitrates concurrent duplicates. The first committed content and
timestamp are retained. Reads query current committed records instead of a process-local cache.

Compression saves extracted facts as one transaction. An exception prevents the caller from discarding
source entries, so failed compression can be retried. Storage failures are explicit tool failures. On reads
or startup they produce a warning and disable persistent memory while allowing session-only work.
Transient failures can be retried on a later operation. Lock waits and snapshot copying are bounded.

Schema creation and legacy JSON import commit in one transaction. A metadata marker prevents later
reimports. Both the legacy object and list formats are supported. The complete input is validated,
original timestamps and first-occurrence deduplication are preserved, and the JSON is never modified.
Unknown database versions and malformed legacy data are rejected instead of reset.

After a fact transaction, create a verified SQLite backup in a temporary file and atomically publish it
as `.sqlite3.bak`. Serialize snapshot publication with writers so an older snapshot cannot replace a
newer generation. A backup error is reported without misrepresenting the already committed primary save.
New or migrated databases also receive a snapshot. Backups are for recovery from corruption or loss;
they do not provide revision history. The store is intended for local filesystems.

`python -m mikucli.memory backup|restore` operates without an API key. Restore is offline: all sessions
using that workspace must be stopped. Validate and flush the replacement, preserve the previous database
and recovery journals, then atomically replace the primary. A durable pending marker prevents startup
from treating an interrupted recovery as a new empty store. Explicit `restore --resume` completes recovery.
Recovery retains legacy JSON and previous files for inspection; it never attempts speculative JSON salvage.

## Verification

Regression tests cover independent readers, concurrent worker threads and separate processes, shared-key
deduplication, original timestamps, failed statements and commits, lock contention, killed transactions,
killed migration, malformed inputs, corrupt databases, failed backups, and interrupted restore.
CLI tests verify usable session-only work, explicit failed save tools, preserved compression inputs,
and credential-free backup and restore.
