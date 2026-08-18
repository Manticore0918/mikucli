# Short code-review task

Copy and paste this prompt into a normal single-agent mikucli session:

```text
Review showcase/code_review/inventory.py against its documented business invariants. Identify the highest-impact
correctness issue, run only the targeted showcase/code_review tests to confirm it, make the smallest safe fix, and
run the same tests again. Do not change unrelated files. In the final answer, explain the failure mode and why the
new validation belongs at this boundary.
```

Suggested targeted command:

```powershell
python -m unittest discover -s showcase\code_review\tests -v
```

