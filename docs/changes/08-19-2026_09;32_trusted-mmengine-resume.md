# Trusted MMEngine resume

The first adaptation segment could not load the completed 5,000-iteration
detector checkpoint because PyTorch's restricted `weights_only=True` loader
rejects MMEngine `HistoryBuffer` objects stored in the message hub.

## Trusted checkpoint boundary

- Detector resume now requires exactly one completed
  `ada_fnp.train_detector` artifact recorded in the run state.
- The checkpoint must resolve below the run-local `checkpoints` directory.
- Artifact type, producer stage, artifact ID, relative path, and SHA256 are
  validated before full pickle loading is allowed.
- `weights_only=False` is used only after those checks. Unrecorded, moved, or
  modified checkpoint files are rejected before deserialization.
- Shape/key, optimizer, parameter-scheduler, and global-iteration validation
  remains mandatory after loading.

## Exact segmented iteration resume

- Future detector checkpoints explicitly store both `iter` and
  `global_iteration` at the completed global step.
- Checkpoints created before this correction used MMEngine's default
  by-iteration `iter + 1` metadata. That single known legacy offset is
  normalized in memory; the original checkpoint and its recorded SHA256 are
  not modified.
- Detector stages use `ADAODSegmentedIterBasedTrainLoop`. It preserves the
  global optimizer/scheduler iteration while preventing MMEngine from reading
  thousands of discarded batches from the newly constructed next-stage
  dataloader.
- A checkpoint that exists without a completed artifact is not selected for
  resume. The previous verified stage checkpoint is used instead.

## Validation

- Regression coverage includes a real MMEngine `HistoryBuffer`, artifact
  tampering, strict model/reproducibility state, legacy iteration
  normalization, and logical dataloader skipping.
- The relevant repository suite completed with 209 passed and one optional
  real-layout test skipped.
- The existing seed-0 run state and its 5,000-iteration checkpoint were not
  rewritten. It remains resumable from the failed 5,000-to-10,000 stage.
