# Remove unused and duplicate runtime code

## Summary

Removed project code that was not reachable from the ADAOD CLI, method
manifest, MMEngine config, MMDetection registry, or ADA-FNP stage plan. Shared
runtime responsibilities now have one implementation instead of parallel
helpers.

## Removed unused paths

- Removed the root `.gitkeep` placeholder.
- Removed `methods/ada_fnp/training/teacher.py`; detector initialization already
  performs the teacher copy at the 5k phase boundary.
- Removed `methods/common/engine/plan.py`; its validator was never called and
  encoded an executor name not used by the concrete plan.
- Removed `tools/internal/resolve_config.py` and the duplicate `--dry-run` CLI
  path.
- Removed unused MC collection, pseudo-label projection, localization-score,
  annotation-oracle, phase-sampler, torchvision VGG conversion, NMS mapping,
  and GradientReversal module-wrapper code.
- Removed unused package re-exports and internal `__all__` declarations that
  caused unrelated modules to load eagerly.
- Removed test-injection-only execution services, CPU backend branches,
  downloader injection, legacy one-argument stage executors, and unused plan
  resume-policy metadata.

## Consolidated responsibilities

- `methods/common/artifacts.py` now owns the sole SHA256 implementation and
  run-contained artifact path resolution.
- `ArtifactStore` now creates references for both JSON and existing files;
  ADA-FNP executors no longer construct duplicate `ArtifactRef` objects.
- ADA-FNP round pool, target-labeled, and target-unlabeled paths are defined in
  `methods/ada_fnp/execution/paths.py`.
- Total budget resolution is shared between plan construction and pool
  initialization.
- FNPM iteration limits come from the phase contract, while its learning rate,
  matching threshold, and maximum detections come from the resolved method
  config.
- Acquisition normalization constants, empty-detection score, discriminator
  epsilon, detector regression mode, branch batch sizes, EMA momentum, and MC
  settings now use the resolved method config instead of parallel literals.

## Compatibility

The resolved configuration fingerprint for the existing C-to-F seed-0 run is
unchanged, so `--resume` continues to target the same run state. The MMEngine
config still exposes the model, stage overrides, acquisition dataset, train
loop, and evaluator required by the execution backend.

## Verification

- Parsed all project Python files with the standard-library AST without
  importing them: 85 files parsed.
- Confirmed there are no top-level functions or classes whose name appears
  only at its definition.
- Confirmed removed symbols have no current source or documentation references.
- Confirmed the existing seed-0 resolved-config fingerprint still matches.
- Confirmed the MMEngine config parses with only the runtime stage keys.
- Confirmed no Python or pytest cache directories were created.
- Ran `git diff --check` after cleanup.

No test, smoke, model build, training, inference, or CUDA operation was run.
