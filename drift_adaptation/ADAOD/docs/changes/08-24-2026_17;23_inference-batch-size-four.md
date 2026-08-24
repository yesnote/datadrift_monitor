# Inference batch size four

## Summary

- Added one resolved `inference` configuration mapping for acquisition and
  final evaluation batch sizes.
- Set both inference batch sizes to 4.
- Connected acquisition scoring and MMDetection validation/test dataloaders
  to the resolved values so they are recorded in `resolved_config.json` and
  its fingerprint.

## Compatibility

Existing runs retain the configuration written when they started. The new
batch sizes apply only to newly started runs.

## Validation

- Parsed the modified Python sources with `ast.parse` without importing the
  training stack.
- Confirmed the old acquisition batch-size literal is absent from the runtime
  path.
- Confirmed `git diff --check` reports no whitespace errors.
- No test, smoke, model, dataset, or experiment command was run.
