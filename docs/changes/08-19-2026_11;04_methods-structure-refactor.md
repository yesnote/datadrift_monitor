# Methods structure refactor

## Scope

The `methods` tree was reorganized so that filenames, classes, configuration
keys, and stage executors describe their actual responsibilities. Duplicate
artifact, path, phase, model, and training helpers were consolidated. The
runtime behavior and the Cityscapes-to-Foggy ADA-FNP experiment contract were
preserved.

## Shared modules

- Consolidated atomic file writes, canonical JSON, SHA256 helpers, contained
  run paths, and `ArtifactStore` in `methods/common/artifacts.py`.
- Moved verified HTTPS asset preparation to
  `methods/common/external_assets.py`.
- Consolidated score and selection serialization in
  `methods/common/acquisition/score_artifacts.py` with
  `AcquisitionArtifact` and `AcquisitionArtifactRecord`.
- Replaced the single Cityscapes module and separate annotation helper with
  `methods/common/data/cityscapes/layout.py`, `conversion.py`, and `reveal.py`.
- Added `methods/common/engine/executor_loader.py`. A method manifest now
  names an explicit `executor_module`; the loader obtains that module's
  `create_executor_registry` factory without a method-name branch.
- Renamed the PT-specific metric to `Detectron2PascalVocMetric` in
  `methods/common/mmdet/metrics/detectron2_voc_metric.py` and renamed the
  pretrained checkpoint definition to
  `methods/common/mmdet/models/backbones/vgg16_caffe_checkpoint.py`.

## ADA-FNP modules

- Added `schedule.py` as the single source for detector segments, acquisition
  milestones, predictor iteration counts, and budget resolution.
- Consolidated domain adaptation in `models/domain_adaptation.py` and exposed
  `AdaFnpDomainDiscriminator`.
- Renamed the detector components to `AdaFnpDetector` and
  `AdaFnpDetectorBranch`.
- Renamed the MC Dropout RoI head to
  `AdaFnpMonteCarloDropoutRoIHead`.
- Replaced the abbreviated predictor filenames and symbols with
  `models/false_negative_predictor.py`,
  `training/false_negative_matching.py`, and
  `training/false_negative_training.py`; the model class is
  `FalseNegativePredictor`.
- Moved pseudo-label behavior to `training/pseudo_labeling.py` and the PT
  strong transform to `probabilistic_teacher_augmentation.py`, registered as
  `ProbabilisticTeacherStrongAugmentation`.
- Consolidated acquisition geometry, records, and component calculation in
  `acquisition/scoring.py`; MC Dropout context remains in
  `acquisition/mc_dropout.py`.
- Split execution responsibilities into `execution/stages.py`,
  `mmdet_backend.py`, `mmdet_config.py`, `mmdet_checkpoints.py`, and
  `run_files.py`.
- Made the resolved ADAOD mapping the source of the MMDetection optimizer,
  schedule, batch-size, detector, domain, EMA, MC Dropout, and pseudo-label
  stage settings.
- Removed the duplicate root 40k schedule. The ADA-FNP config now constructs
  its optimizer and schedulers directly from the method defaults, while
  `schedule.py` derives acquisition milestones, round count, and maximum
  iteration from the detector segment table.
- Limited Caffe VGG initialization to the first detector segment; full
  checkpoint builds no longer deserialize and immediately overwrite it.
- Removed intra-round false-negative predictor checkpointing and full-teacher
  CPU hashing. Interrupted predictor rounds restart with their fresh optimizer
  instead of paying this runtime cost.
- Removed the user-facing interrupted-run resume path and completed-stage
  skipping. A run now starts only in an empty directory; the internal detector
  segment continuation remains because it preserves the single 40k optimizer
  and parameter-scheduler trajectory.
- Disabled MMEngine's implicit custom-module import during config parsing;
  execution performs the import exactly once. Helper config objects are
  deleted after the exported stage mappings are built.
- Replaced abbreviated stage executor keys with descriptive `ada_fnp.*` keys
  for asset preparation, dataset preparation, detector training,
  false-negative predictor training, pool scoring, selection, annotation
  reveal, and final evaluation.

## Configuration and compatibility

- The manifest API is version 2 and declares
  `methods.ada_fnp.execution.stages` as its `executor_module`.
- Run state is schema version 2 and stores content identifiers through the
  generic `artifact_ids` mapping.
- Configuration and MMDetection registration use the descriptive class and
  component names listed above.
- Existing runs created with the former run-state schema or abbreviated
  registry/executor names are intentionally incompatible. They must be rerun
  from the beginning in a fresh run directory; old outputs are not modified.

## Validation boundary

Only static source inspection and Git diff validation are used for this
refactor. No test suite, smoke workflow, model build, experiment, or generated
validation artifact is created.

- AST parsing passed for all 80 Python files under `methods`, `configs`, and
  `tools`; the static local-import graph has no missing target.
- MMEngine-style base composition exposes the expected detector, RoI head,
  stage overrides, optimizer, and 40k schedule without leaking data helpers.
- Resolved-config projection was inspected with non-default optimizer, batch,
  dropout, and EMA values; every value reached its MMDetection stage field.
- The plan contains 29 stages, its eight executor keys match the eight
  registered stage executors, and the six detector segments cover 0 through
  40k without a gap.
- The current-document/runtime stale-name scan found no retired path or
  registry name. The paper abbreviation appears once, only where it is mapped
  to `FalseNegativePredictor`.
- No test, smoke, pytest, bytecode, or cache artifact exists in the project
  source roots. `git diff --check` passed.
