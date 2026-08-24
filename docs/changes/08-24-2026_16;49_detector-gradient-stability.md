# Detector gradient stability

## Problem

The first Cityscapes-to-Foggy run remained numerically stable through
iteration 7,400, but its detector losses increased from about 1.30 to about
27.98 million by iteration 7,450 and became NaN by iteration 7,500. Training
continued to iteration 10,000, so false-negative predictor training reported
the delayed symptom `features must be finite` when it consumed the invalid
teacher checkpoint.

## Changes

- Added PT's global gradient-norm clipping with maximum norm 10 and norm type
  2 to the canonical ADA-FNP training configuration.
- Projected the resolved clipping values into every MMDetection detector stage
  so the saved configuration and runtime optimizer wrapper cannot diverge.
- Enabled `error_if_nonfinite` so an invalid gradient stops training before
  `optimizer.step()` rather than contaminating later checkpoints.
- Updated the current reproduction documentation and decision record.

## Run handling

The existing iteration-10,000 detector checkpoint is numerically invalid and
must not be resumed. A corrected experiment must start from a new, empty run
directory.

## Validation

Only static validation was performed, following the repository policy against
test and smoke artifacts. No model, dataset, or experiment output was created
or modified.
