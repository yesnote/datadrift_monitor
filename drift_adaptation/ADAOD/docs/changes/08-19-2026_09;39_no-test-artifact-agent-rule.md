# Add no-test-artifact agent rule

## Summary

Updated the repository-wide agent instructions so future implementation work
does not add or retain project-owned tests, smoke workflows, test-only
dependencies, or generated validation artifacts unless the user explicitly
requests them.

## Rules added

- Do not add `tests/`, `test_*.py`, pytest fixtures, or test-only dependencies.
- Do not create or run smoke, dry-run, synthetic, benchmark, or disposable
  experiment workflows without an explicit request.
- Do not write validation runs under `work_dirs/.smoke/`.
- Remove Python, pytest, and coverage caches if an explicitly permitted command
  creates them.
- Preserve real experiment outputs when cleaning temporary artifacts.
- Prefer source inspection, configuration checks, Git diff checks, and the
  actual user-requested runtime path for validation.

## Verification

The instruction and documentation diffs were inspected. No test, smoke,
training, or inference command was run.
