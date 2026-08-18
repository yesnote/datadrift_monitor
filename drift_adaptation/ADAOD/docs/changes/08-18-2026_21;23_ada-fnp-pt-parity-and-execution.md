# ADA-FNP PT parity and execution

## Scope

This change corrects the initial C to F foundation against the PT reference
and connects the serial ADA-FNP plan to a concrete MMEngine/MMDetection
execution backend. It supersedes the detector, dataset, junction, evaluator,
and non-dry execution descriptions in the earlier foundation change record;
that earlier record is retained as history.

## Data and local layout

- Moved the expected read-only junction layout beneath
  `data/Cityscapes/{gtFine,leftImg8bit,leftImg8bit_foggy}`.
- Changed the converter to exact PT class matching in PT registry order and
  excluded `*group` objects instead of storing them as crowd annotations.
- Reproduced PT VOC conversion plus Detectron2 loading coordinates and bumped
  the deterministic cache schema.
- Kept the full target oracle separate from the annotation-free target index.
- Added run-local `target_train_unlabeled_pool_NN.json` materialization from
  the committed pool so acquired samples are excluded from later training and
  scoring.

The rebuilt cache contract is 2,975/52,469 source images/annotations,
2,975/52,469 target-oracle images/annotations, 2,975/0 initial unlabeled target
images/annotations, and 500/10,180 validation images/annotations.

## PT-compatible detector and training

- Added automatic, checksum-pinned preparation of the Caffe-converted VGG16
  asset from Zenodo record 4515252. The expected file is 553,433,685 bytes,
  MD5 `433ad40ddbd662d6448e13a6cef812f2`, and SHA256
  `736b4bd0b787438253ea1926f9a02730b2eedbf0e48df243457d17133fe8850e`.
- Consolidated the asset path, URL, SHA256, MD5, and byte size in the canonical
  `methods/common/mmdet/models/backbones/vgg16_caffe.py` specification used by
  configuration and execution.
- Made VGG construction load the convolution weights, verify the SHA256, and
  fail instead of silently using random initialization.
- Matched PT's Caffe BGR preprocessing, 600/1,333 resize, frozen VGG stages,
  anchors, RPN sampling and proposal limits, aligned RoIAlign, 1,024-wide RoI
  fully connected layers with C2 Xavier initialization, and class-specific
  bbox regression.
- Added PT's 400-iteration linear warm-up from factor 0.001 and LR drops at
  30k and 35k.
- Kept FNPM source and selected-target batches at four by dropping incomplete
  dataloader tails before deterministic cycling.
- Moved resize and flip before target weak/strong branching and limited the
  strong-only difference to PT photometric augmentation.
- Updated fixed-proposal MC inference for class-specific bbox regression and
  one post-average multiclass NMS.
- Recorded paper-first weak/strong routing for unlabeled target data only,
  Softplus count regression despite Figure 4's Sigmoid label, and omission of
  PT gradient clipping because ADA-FNP does not specify it.

## Evaluation and execution

- Added `PTVOCMetric` with Detectron2 text quantization, half-open box
  arithmetic, strict `IoU > 0.5`, VOC2012 continuous AP, and percentage-scale
  AP50 output.
- Added a real execution plugin for pretrained and dataset preparation,
  segmented detector training, resumable FNPM optimization, complete current
  pool scoring, deterministic selection, selected-only annotation reveal, and
  final teacher evaluation.
- Added deterministic run directories, saved resolved-config fingerprints,
  explicit `--resume` validation, and `--offline` asset behavior.
- Added pre-resume validation for exact detector model keys and tensor shapes,
  optimizer and parameter-scheduler sections, and expected global iteration;
  inference checkpoint loading is strict.
- Expanded score artifacts to retain raw and normalized components,
  source-domain probability, detection count, and final score; selection
  verifies the stored calculations before use.
- Bound every scoring result to the stable sample identity resolved from its
  manifest image ID instead of relying on dataloader iteration order.

## Validation boundary

Unit and real-layout validation are coordinated in the final repository-wide
integration pass and must be reported from commands actually executed there.
This documentation was checked against the current implementation contracts.

The environment checker now bootstraps the repository root for direct-script
execution and uses MMCV 2.1's RoIAlign module API. It passed the pinned stack,
repository-local MMDetection import, CUDA 11.8, GPU NMS, and GPU RoIAlign
forward/backward checks on an NVIDIA GeForce RTX 3090.

The official C-to-F config now parses through MMEngine, its full
`ADAFNPDetector` builds on CUDA, and a real 4-source plus 4-target initial batch
passes preprocessing and direct loss/backward with finite loss. Initial-stage
datasets now omit the absent labeled-target branch, while adaptation-stage
datasets retain the full four-branch schema. Strong-augmentation arrays are
materialized as writable contiguous `uint8` buffers before tensor conversion.

The relevant unit and integration suite initially passed with 201 tests and
one optional real-layout skip. Running the real-layout Cityscapes tests
explicitly passed all seven tests. The first one-iteration MMEngine Runner
smoke then exposed the deterministic RPN compatibility issue described below.
The MMDetection/OpenCV port does not claim pixel- or RNG-trajectory identity
with PT's Detectron2/PIL runtime.

## Deterministic CUDA RPN compatibility

The blocking RPN assignment was reduced to PyTorch 2.0.1 Windows CUDA scalar
advanced indexing under deterministic algorithms. The vendored MMDetection
line now assigns an equal-shaped tensor of ones instead. This is a semantic
no-op and keeps deterministic mode enabled. A focused CUDA regression exercises
the real `RPNHead._get_targets_single` path with 64 sampled positives.

The next Runner gate exposed PyTorch's deterministic CuBLAS workspace
requirement. `tools.run_adaod` now sets the larger `:4096:8` workspace mode
before importing the execution stack while preserving an explicit caller
value.

With both compatibility requirements applied, the official C-to-F Runner
completed iteration 1 and wrote `iter_1.pth`. The checkpoint contains optimizer
and parameter-scheduler state; a fresh Runner loaded it and completed iteration
2. This closes the execution blocker, but it does not constitute a completed
40k experiment or scientific AP50 reproduction.

Final validation completed with 203 relevant unit/integration tests passing and
one optional real-layout test skipped in the general suite. The real Cityscapes
layout suite passed all seven tests when explicitly enabled. The environment
checker, ADA-FNP dry-run, Python compilation, and `git diff --check` also
passed. The only vendored MMDetection difference introduced for this blocker
is the documented RPN bbox-weight assignment change.
