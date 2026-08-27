# Zero-budget UDA and checkpoint evaluation

## Summary

- Added an explicit `--budget-percent 0` execution path for the ADA-FNP C to F
  UDA result.
- Added teacher AP50 evaluation after detector iterations 5k, 10k, 15k, 20k,
  25k, and 40k for both zero-budget and active runs.
- Kept the final `artifacts/evaluation.json` interface while adding one
  iteration-tagged artifact per checkpoint.

## Execution behavior

The zero-budget plan contains 14 stages: two preparation stages followed by
six detector-training and six teacher-evaluation stages. It does not create
false-negative predictor, acquisition-score, selection, reveal, or
target-labeled artifacts. The 0-to-5k initialization is unchanged. Later
segments use source and target-unlabeled batches, pseudo-label classification,
and teacher EMA while preserving the continuous optimizer and learning-rate
schedule.

Active runs contain 34 stages because a checkpoint evaluation follows each of
the six detector segments. Evaluation resolves and verifies the checkpoint
for the requested iteration rather than accepting the latest checkpoint.
Intermediate MMEngine outputs are isolated under
`mmengine/evaluations/iter_NNNNN`, and checkpoint metrics are written to
`artifacts/evaluations/detector_NNNNN.json`.

Evaluation preserves and restores Python, NumPy, Torch CPU, and Torch CUDA RNG
states. The terminal prints one permanent AP50 summary line per checkpoint
without replacing the reusable progress bar.

## Validation boundary

Validation for this change is static and does not execute model training,
CUDA inference, pytest, or a smoke experiment. Existing datasets, checkpoints,
and experiment outputs were not modified.
