# mikucli recording showcase

This folder contains two deliberately small review fixtures for recording a short mikucli demo. The fixtures use
only Python's standard library, and their tests are intentionally red until the agent completes the requested
review.

The showcase tests are outside the repository's normal `tests/` directory, so they do not affect the main mikucli
test suite unless they are invoked explicitly.

## Recording 1: single-agent review and approvals

Start mikucli from the repository root:

```powershell
mikucli --workspace D:\Personal_Projects\mikucli
```

Paste the prompt from [code_review/TASK.md](code_review/TASK.md). A good recording should show:

1. automatic low-risk `list_files` or `read_file` calls;
2. a `run_shell` approval before the targeted tests execute;
3. a proposed diff and `write_file` approval;
4. a final targeted test run and concise explanation of the defect.

Keep this take around 20–30 seconds after trimming or accelerating model wait time.

## Recording 2: planner, workers, and reviewer

Start a fresh interactive session, then enter:

```text
/team
```

Paste the prompt from [team_review/TASK.md](team_review/TASK.md). The task contains two independent review surfaces
so the planner can assign one to each worker. Keep enough of the terminal visible to capture these prefixes:

```text
[planner-1]
[worker-1]
[worker-2]
[reviewer-1]
```

Approve the serialized writes and targeted test commands as they appear. For the cleanest recording, stop after the
orchestrator prints its final execution summary.

## Baseline commands

These commands are expected to fail before the demo fixes are applied:

```powershell
python -m unittest discover -s showcase\code_review\tests -v
python -m unittest discover -s showcase\team_review\tests -v
```

After a successful recording, both commands should pass.

If you commit the fixture in its intentionally failing state, you can reset it between recordings with:

```powershell
git restore -- showcase/code_review showcase/team_review
```

