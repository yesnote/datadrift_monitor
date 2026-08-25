# Reproducibility

ADAOD distinguishes configuration-level implementation parity from scientific
reproduction. The implementation below encodes the paper and PT reference
choices, but scientific parity requires completing the real GPU experiment and
comparing AP50 results.

## C to F contract

| Item | Implemented value |
| --- | --- |
| source | Cityscapes train, 2,975 images |
| target pool | Foggy Cityscapes train, beta 0.02, 2,975 images |
| evaluation | Foggy Cityscapes val, beta 0.02, 500 images |
| categories | `truck, car, rider, person, train, motorcycle, bicycle, bus` |
| group labels | excluded exactly; not treated as a ninth class or crowd GT |
| input size | short edge 600, long edge at most 1,333 |
| normalization | Caffe BGR mean, unit standard deviation |
| detector duration | 40k iterations; acquisitions at 5k, 10k, 15k, 20k, 25k |
| optimizer | SGD, LR 0.02, momentum 0.9, weight decay 0.0001 |
| schedule | linear warm-up 0--400 from 0.001; drops at 30k and 35k |
| teacher | initialized from student at 5k; EMA decay 0.9996 thereafter |
| false-negative predictor | Softplus count output; fresh 2k-step optimization per round, LR 1e-4 |
| pseudo-label filter | RoI bbox-delta variance at most 0.1 and foreground confidence at least 0.5 |
| evaluation | PT-compatible VOC2012 continuous AP at strict IoU greater than 0.5 |

Source and selected-target supervision receive PT strong photometric
augmentation according to Supplementary Equations S2 and S9. Unlabeled target
data uses weak-teacher and strong-student views. `FalseNegativePredictor` uses
Softplus because a
Sigmoid, despite its label in Figure 4, cannot regress raw false-negative
counts greater than one. Detector optimization follows PT's global
gradient-norm-10 clipping and rejects non-finite gradients before the
optimizer step. This prevents a numerically invalid detector from being
carried into later false-negative prediction stages.

The MMDetection port matches PT's configured resize and augmentation
distribution, but its OpenCV resize implementation and random-number
consumption are not guaranteed to be pixel- or seed-trajectory-identical to
Detectron2/PIL. This remains part of the scientific validation boundary.

The converter reproduces PT's effective VOC-to-Detectron2 coordinates: it
clips coordinates only when they cross image boundaries, subtracts one from
the lower x/y bounds, and retains the upper bounds. The generated COCO boxes
therefore represent zero-based, half-open xyxy coordinates. Exact class-label
matching excludes `*group` regions. The prepared cache contract is 52,469
annotations for source train, 52,469 for the target oracle, zero for the
unlabeled target index, and 10,180 for target validation.

`Detectron2PascalVocMetric` avoids MMDetection's legacy inclusive `+1`
arithmetic. It also
matches Detectron2's Pascal text quantization (bbox to 0.1, confidence to
0.001), uses a strict `IoU > 0.5` match, continuous-area interpolation, and
reports AP50 as a percentage.

## Active-pool isolation and artifacts

The target oracle is not referenced by a training or scoring dataset. Round
zero starts with all 2,975 target images in the committed unlabeled pool. Each
stage materializes an annotation-free run-local manifest containing exactly
the current unlabeled sample IDs; selection commits a new pool state before
the reveal stage writes a selected-only labeled manifest. Thus acquired images
are removed from subsequent unlabeled training and scoring.

Every acquisition-score record stores raw false-negative, localization,
entropy, and diversity values; their normalized values; source-domain
probability; detection count; and the final product. Artifact metadata records
the MC pass count, RoI bbox-delta variance space, and both pseudo-label
thresholds. Stored scores are
recomputed and checked before selection. Empty-detection images have final
score zero.

The run records its resolved config and fingerprint, atomic schema-version-2
stage state, content-addressed artifacts, checkpoints, selections, and
evaluation metrics. Shared atomic I/O and hash handling live in
`methods/common/artifacts.py`; remote asset verification lives in
`methods/common/external_assets.py`; and the common score/selection schema
lives in `methods/common/acquisition/score_artifacts.py`.
Interrupted user runs are restarted rather than resumed, and ADAOD refuses to
overwrite a nonempty run directory. Internal detector segment continuation
compares model keys and tensor shapes exactly, requires optimizer and
parameter-scheduler sections, and accepts only the expected global iteration
before handing the preceding checkpoint to MMEngine. Inference loads model
state strictly. Randomness uses the configured seed with
deterministic mode enabled and cuDNN benchmarking disabled. The ADAOD CLI also
sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` before importing the execution stack,
unless the caller already selected a CuBLAS workspace mode. This is required
by PyTorch deterministic CUDA matrix multiplication.

## Current validation boundary

The concrete execution modules connect detector training, false-negative
predictor training, pool scoring, selection, annotation reveal, and teacher
evaluation to MMEngine/MMDetection. `schedule.py` defines the experiment
timeline; `execution/stages.py` owns stage orchestration;
`execution/mmdet_backend.py` owns model work; `execution/mmdet_config.py` owns
stage configs; `execution/mmdet_checkpoints.py` owns strict detector
checkpoints; and `execution/run_files.py` owns run-local paths and manifests.

The pinned environment and vendored-MMDetection compatibility changes predate
this structural refactor. This refactor is validated statically only: it does
not run a model, CUDA workflow, synthetic workflow, or abbreviated experiment.
The resolved ADAOD configuration is explicitly projected onto every
MMDetection stage config, and the Caffe VGG asset is loaded only for the
initial 0-to-5k model build. Continuation, predictor, scoring, and evaluation
builds rely on their strict full detector checkpoint instead.

Manifest API version 2, run-state schema version 2, and the descriptive
registry/executor names intentionally reject older runs. Reproducibility runs
created before this refactor must be restarted in a fresh run directory.

An end-to-end 40k run and the paper's AP50 values remain unvalidated. The
implementation must not be described as a scientific reproduction until these
gates are closed. At minimum, report source-only, 0 percent, 1 percent, and 5
percent C to F results plus component ablations over three seeds before making
that claim.
