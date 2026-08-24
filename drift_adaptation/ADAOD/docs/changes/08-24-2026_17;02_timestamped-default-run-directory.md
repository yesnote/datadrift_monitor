# Timestamped default run directory

## Change

ADAOD now creates the default experiment directory at

```text
work_dirs/runs/<method>/<scenario>/<detector>/seed-<seed>/MM-DD-YYYY_HH_mm
```

The timestamp uses the machine's local time and a 24-hour clock. This keeps
multiple runs for the same method, scenario, detector, and seed separate
without requiring `--run-directory` on every command.

An explicit `--run-directory` remains an exact override; ADAOD does not append
a timestamp to a user-selected path. If two default executions start within
the same minute, the existing nonempty-directory protection rejects the
second execution instead of overwriting the first.

## Validation

The changed Python source was parsed with `ast` and `git diff --check` was
run. No experiment, test, smoke workflow, cache, or output directory was
created.
