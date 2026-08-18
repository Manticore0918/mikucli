# Parallel team-review task

Enter `/team`, then copy and paste this prompt:

```text
Review showcase/team_review as two independent workstreams. Assign one worker to pricing.py and its targeted tests,
and the other worker to inventory.py and its targeted tests. Each worker must check the documented invariants,
identify the boundary-validation defect, make the smallest safe fix, and run only its related test file. Do not
combine the modules into one implementation step and do not change unrelated files. Have the reviewer verify both
fixes and report whether the two business invariants now hold.
```

Suggested targeted commands:

```powershell
python -m unittest showcase.team_review.tests.test_pricing -v
python -m unittest showcase.team_review.tests.test_inventory -v
```

