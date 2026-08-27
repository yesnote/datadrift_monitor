# Run artifacts

Generated files live at `work_dirs/<method>/<scenario>/<run>`. Each run stores
an immutable resolved configuration and plan manifest, mutable atomic state,
checkpoints, round score shards, merged scores, selections, pool transitions,
and final evaluation metrics.

ADA-FNP evaluates its teacher and AADA evaluates its detector immediately
after iterations 5k, 10k, 15k, 20k, 25k, and 40k. The corresponding artifacts are
`artifacts/evaluations/detector_05000.json` through
`artifacts/evaluations/detector_40000.json`; each stores the exact checkpoint
iteration and AP50. The 40k metric is duplicated in
`artifacts/evaluation.json` as the stable final-result interface. Evaluation
logs are separated under `mmengine/evaluations/iter_NNNNN`.

`methods/common/artifacts.py` is the single implementation of canonical JSON
bytes, atomic bytes/JSON writes, SHA256 calculation, repository-contained run
paths, and `ArtifactStore`. JSON artifacts are key-sorted, written to a
temporary sibling, flushed, and atomically replaced. Their SHA256 is recorded
in an artifact reference.

`methods/common/acquisition/score_artifacts.py` defines the immutable,
sample-keyed `AcquisitionArtifact` and `AcquisitionArtifactRecord` schema used
for both round scores and selections. Raw and normalized score fields stay in
one record instead of using method-specific file readers. External downloads
use `methods/common/external_assets.py`, which verifies the pinned checksum
before atomically installing an asset.

ADA-FNP score metadata records the MC pass count, `roi_bbox_delta` variance
space, localization threshold, and confidence threshold. Detector scalar logs
also retain pseudo-label candidate, variance-pass, confidence-pass, and final
keep counts without adding them to terminal progress output.

Run-local pool state, selected-only labeled manifests, annotation-free
unlabeled manifests, and exact-iteration checkpoint lookup belong to
`methods/common/execution/run_files.py`. Detector checkpoint structure and
strict segment-continuation validation belong to
`methods/common/execution/mmdet_checkpoints.py`. Checkpoints preserve model,
optimizer, scheduler, and global iteration for segmented detector training.
The false-negative predictor stores only completed-round model weights because
an interrupted predictor round is rerun from its beginning.

Converted dataset annotations are generated beneath
`work_dirs/.dataset_cache`; no generated annotation is written into `data`.
Layout validation, conversion, and reveal are separate operations in
`methods/common/data/cityscapes/{layout,conversion,reveal}.py`.

Compact terminal progress does not replace experiment records. The MMEngine
`LoggerHook` continues to write its complete 50-iteration tag set to the
timestamped `.log` and `vis_data/scalars.json`; the run root still contains
`resolved_config.json`. Only the non-file console handler suppresses routine
INFO output. ADA-FNP initialization records source detector and domain losses,
while adaptation records source, domain, labeled-target, and strong
unlabeled-target classification values.

Run state is schema version 2 and records artifact identifiers in the generic
`artifact_ids` mapping. Schema-1 run state and run artifacts produced under
former registry/executor names are intentionally incompatible; restart them
in a fresh run directory rather than editing their files.
