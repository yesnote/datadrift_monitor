# ADA-FNP Cityscapes-to-Foggy foundation

## Scope

This change establishes the first ADAOD implementation slice for ADA-FNP on
Cityscapes to Foggy Cityscapes at beta 0.02. The only supported detector is a
BN-free VGG16 Faster R-CNN. Reference repositories remain read-only and are
not runtime dependencies.

## Repository structure

- Added the root `configs`, `data`, `docs`, `methods`, `mmdet`,
  `requirements`, `tools`, and `work_dirs` areas.
- Kept reusable active-domain-adaptation state, artifact, selection, dataset,
  and framework-extension code in `methods/common`.
- Kept ADA-FNP-specific model, training, acquisition, config, and tests under
  `methods/ada_fnp`.
- Vendored pristine MMDetection 3.3.0 at commit
  `44ebd17b145c2372c4b700bfb9cb20dbd28ab64a`; project extensions register
  externally without modifying or importing project code from `mmdet`.
- Added method discovery through `methods/*/manifest.py`; the CLI and common
  runner contain no method-name dispatch.

## Data and artifacts

- Created local `data/gtFine`, `data/leftImg8bit`, and
  `data/leftImg8bit_foggy` junctions to the user-provided dataset directories.
  Runtime configuration refers only to these repository-relative paths.
- Added deterministic Cityscapes polygon-to-COCO conversion for source train,
  Foggy train unlabeled, a separate Foggy train oracle, and Foggy validation.
- Enforced a 2,975-image source train set, 2,975-image Foggy beta 0.02 target
  pool, and 500-image Foggy beta 0.02 validation set.
- Generated cache manifests below `work_dirs/.dataset_cache`, including input
  and output hashes. Target-unlabeled records contain no annotations, and only
  committed selections can be materialized into the target-labeled manifest.
- Added immutable sample identities, active-pool invariants, deterministic
  five-round budget allocation, canonical artifacts, atomic writes, hash
  verification, and resumable stage state.

## ADA-FNP implementation

- Added BN-free VGG16 through conv5_3 with no pool5, a 7-by-7 4096-dimensional
  fc6/fc7 RoI head, two dropout sites, torchvision VGG weight mapping, and a
  pure gradient-reversal layer.
- Added the Progressive-DA image discriminator, source and labeled-target
  supervised routing, target adversarial routing, and classification-only
  unlabeled losses.
- Added the student/teacher branch, exact student-to-teacher initialization,
  EMA configuration, fixed-proposal teacher MC inference, weak-to-strong
  pseudo-label projection, and the PT strong photometric augmentation.
- Added TIDE-style false-negative targets at IoU 0.5, FNPM construction and
  isolated 2,000-step per-round optimization with exact resume state.
- Added raw acquisition components for FN count, normalized box variance,
  foreground entropy without foreground renormalization, and domain ratio;
  added Equation 10 normalization, deterministic score fusion, and selection.
- Added the five acquisition milestones at 5k through 25k and continuous 40k
  detector plan. The 1-percent budget is 30 images as 6 per round; the
  5-percent budget is 149 images as 30, 30, 30, 30, and 29.

## Subagent work split

- Framework owner: MMDetection provenance, VGG16 and RoI integration,
  detector/teacher wrapper, fixed-proposal inference, and PT augmentation.
- Common-core owner: sample identity, pool, artifacts, Cityscapes conversion,
  reveal isolation, phase contracts, and FNPM optimization.
- Configuration owner: catalogs, shared MMDetection configs, ADA-FNP overlay,
  branch pipelines, stage overrides, and config tests.
- Main integration: contracts, manifest discovery, CLI, junctions, shared-I/O
  deduplication, cross-wave review, full validation, and documentation.

## Validation performed

- `python -m pytest -p no:cacheprovider methods/common/tests methods/common/mmdet/tests methods/ada_fnp/tests tools/tests -q -rs`
  completed with 158 passed and 4 skipped.
- With `ADAOD_VALIDATE_REAL_CITYSCAPES=1`,
  `python -m pytest -p no:cacheprovider methods/common/tests/test_cityscapes.py -q -rs`
  completed with 6 passed.
- `python -m tools.run_adaod --list-methods` discovered only `ada-fnp`.
- The 1-percent and 5-percent `--dry-run` plans resolved successfully with
  exact cumulative budgets of 30 and 149.
- `python -m compileall -q configs methods tools` completed successfully.
- `git diff --check` reported no whitespace errors.
- A duplicate-helper scan found one shared implementation each for hashing,
  atomic writes, budget splitting, and normalized box conversion.
- The prepared real-data cache contains 2,975 source images with 56,128
  annotations, 2,975 target-unlabeled images with zero annotations, a separate
  2,975-image oracle with 56,128 annotations, and 500 validation images with
  10,717 annotations.

## Remaining gate

The current interpreter has PyTorch 2.8.0 CPU-only and lacks MMCV and
MMEngine. `python tools/check_environment.py --allow-cpu` therefore fails as
intended at the PyTorch version gate. Four tests were skipped for the optional
real-layout flag or missing OpenMMLab dependencies. The real layout was tested
separately; the remaining three require the pinned CUDA environment.

Full MMDetection model construction, CUDA NMS and RoIAlign, end-to-end runner
execution, pretrained VGG loading, full training, and scientific AP50 parity
have not yet been validated. The non-dry CLI remains deliberately disabled
until those gates and concrete stage executors are complete.
