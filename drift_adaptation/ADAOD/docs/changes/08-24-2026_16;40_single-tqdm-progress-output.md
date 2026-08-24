# Single tqdm progress output

## Summary

ADAOD now routes interactive execution through one reusable `tqdm` line.
MMEngine configuration, environment, hook, and 50-iteration metric output are
no longer repeated in the terminal. The non-file MMEngine handler is limited
to errors while the timestamped file handler and scalar visualizer remain at
their existing detail level.

## Implementation

- Added the method-neutral `ProgressReporter` and MMEngine progress adapter.
- Connected all 29 stages, detector train/test loops, false-negative predictor
  optimization, pool scoring, verified downloads, and Cityscapes preparation.
- Added `tqdm>=4.65,<5` as a direct runtime requirement.
- Kept total loss as the only training scalar shown in the progress line.
- Required the six initialization metrics and all thirteen adaptation metrics
  before accepting detector log output.
- Changed successful CLI completion and standalone Cityscapes preparation to
  compact one-line summaries.

## Preserved records

The existing MMEngine `LoggerHook` interval, `LogProcessor`, visualizer,
timestamped `.log`, `vis_data/scalars.json`, `resolved_config.json`, and run
artifacts are unchanged. No loss, acquisition, dataset, schedule, or metric
calculation was changed.

## Validation

- AST parsing succeeded for all fourteen changed or added Python files.
- `git diff --check` passed; Git reported only the existing Windows line-ending
  conversion notices.
- Source inspection found exactly one direct `tqdm` import and confirmed that
  `LoggerHook(interval=50)`, `LogProcessor`, and the visualizer remain enabled.
- The six initialization and thirteen adaptation metric keys were verified in
  the backend contract.
- Seventy-two generated `__pycache__` directories were removed from the
  runtime source roots, leaving no Python/test cache entry there.

No test, smoke, model, or experiment command was run.
