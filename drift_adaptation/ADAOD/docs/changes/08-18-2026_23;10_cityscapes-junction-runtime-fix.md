# Cityscapes junction runtime fix

The production ADA-FNP command failed during `prepare_datasets` because the
execution path guard called `Path.resolve()` on repository-local Cityscapes
junctions. Windows resolved each junction to its external read-only dataset
target, which the guard then misclassified as a configured path escape.

## Implementation

- Kept repository-relative syntax and lexical containment validation for all
  execution paths.
- Added an explicit opt-in for read-only dataset inputs that preserves their
  repository-local junction address instead of dereferencing the target.
- Applied that opt-in only to the source image, target image, and annotation
  roots passed to the Cityscapes-to-Foggy preparer.
- Kept the original resolved-containment check for pretrained assets and all
  other paths that may be written, so a junction cannot redirect those writes
  outside the repository.
- Added regression coverage for allowed dataset junctions, rejected write-path
  junctions, and rejected parent-directory traversal.

## Validation

- The focused execution suite passed with 9 tests.
- Real junction-backed conversion completed with 2,975 source images and
  52,469 annotations, 2,975 target-oracle images and 52,469 annotations,
  2,975 annotation-free target-pool images, and 500 validation images with
  10,180 annotations.
- The full relevant suite completed with 206 passed and one optional
  real-layout test skipped.

The interrupted default seed-0 run remains at `prepare_datasets` with its
verified pretrained stage preserved. It must be continued with `--resume`;
starting the same deterministic run without that flag is intentionally
rejected.
