# AADA under the ADA-FNP protocol

## Summary

- Added AADA as a discoverable method for the existing Cityscapes-to-Foggy
  VGG16 Faster R-CNN experiment.
- Kept ADA-FNP's 40k detector schedule, five active rounds, percentage budget,
  batch sizes, dataset conversion, and checkpoint AP50 evaluation.
- Implemented AADA supervised/adversarial loss routing and raw
  entropy-times-domain-diversity acquisition.
- Moved percentage budget, entropy/domain diversity, Progressive-DA
  discriminator, detector schedule, run files, strict checkpoints, shared
  stages, MMDetection configuration, and runtime code into `methods/common`.
- Added a shared deterministic class-probability RoI head and a shared 40k
  MMDetection schedule instead of copying ADA-FNP implementations.
- Preserved ADA-FNP-only teacher, FN predictor, MC Dropout, pseudo-label, and
  four-component acquisition code inside `methods/ada_fnp`.

## Validation

- Parsed every Python source under `methods`, `configs`, and `tools` with
  Python's AST parser.
- Discovered both `ada-fnp` and `aada` through the manifest registry.
- Composed both method plans and confirmed identical 0-, 1-, and 5-percent
  budget behavior and checkpoint evaluation iterations.
- Evaluated both Python config bodies against the same dataset, detector,
  runtime, and schedule base mappings.
- Built both registered MMDetection model graphs in the pinned Conda
  environment without running a forward pass.
- Did not run tests, smoke experiments, model training, or CUDA inference.
- Added a fail-fast NumPy-major check after the installed NumPy 2.0.2 emitted
  PyTorch 2.0.1's compiled-ABI incompatibility warning during model build.
