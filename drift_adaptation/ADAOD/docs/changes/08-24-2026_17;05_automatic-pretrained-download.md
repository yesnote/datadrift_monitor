# Automatic pretrained checkpoint download

## Change

The `--offline` command-line option was removed from the ADAOD runner and the
standalone pretrained-checkpoint preparer. Pretrained asset behavior is now
unconditional and has one path:

1. Reuse the local checkpoint when its SHA-256 matches.
2. Download it atomically from the pinned HTTPS URL when it is missing.
3. Stop with an explicit error when an existing file has the wrong SHA-256.

The execution context and shared asset API no longer carry an offline flag or
an `allow_download` branch.

## Validation

The changed Python sources were parsed with `ast`, stale current references to
the removed option were checked, and `git diff --check` was run. No network
request, experiment, test, smoke workflow, cache, or output was created.
