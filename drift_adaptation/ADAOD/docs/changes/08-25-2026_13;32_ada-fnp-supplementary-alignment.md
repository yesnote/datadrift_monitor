# ADA-FNP supplementary alignment

## Reason

The ADA-FNP supplementary material makes the uncertainty-guided pseudo-label
contract more explicit than the main paper. It applies both localization and
confidence indicators, computes variance from MC bbox-head outputs, and uses
PT strong augmentation for source and labeled-target supervision.

## Changes

- Added an explicit pseudo-label confidence threshold of 0.5 to the resolved
  method config and MMDetection model config.
- Changed pseudo-label selection to require both mean bbox-delta variance at
  most 0.1 and foreground confidence at least 0.5.
- Changed fixed-proposal MC inference to average class-specific bbox-head
  deltas, compute their unbiased variance, select the mean-probability argmax
  class, decode its mean delta once, and retain the matching variance through
  class-aware NMS.
- Removed image-normalized decoded-box variance from acquisition and
  pseudo-labeling.
- Applied PT strong photometric augmentation to source and selected-target
  supervision while retaining weak-teacher and strong-student target views.
- Added saved pseudo-label candidate/filter counters and acquisition artifact
  metadata for the MC and threshold contract.

## Compatibility

Existing run directories and checkpoints are not modified. Their acquisition
scores use the former normalized decoded-box variance and are not directly
comparable with runs created after this change. New experiments must start in
a new timestamped run directory.

## Validation policy

No test, smoke, cache, or generated experiment artifact was added. Validation
is limited to no-bytecode syntax/config inspection and repository diff review.
