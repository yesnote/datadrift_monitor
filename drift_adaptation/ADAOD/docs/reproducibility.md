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
| FNPM | Softplus count output; fresh 2k-step optimization per round, LR 1e-4 |
| evaluation | PT-compatible VOC2012 continuous AP at strict IoU greater than 0.5 |

Two choices resolve paper/reference ambiguity rather than establish confirmed
equivalence. Only the unlabeled target data receives weak-teacher and
strong-student views; source and selected-target supervision use the weak view
only. This follows ADA-FNP Figure 2 and Equations 13--14 rather than PT's use of
both views for labeled/source supervision. FNPM uses Softplus because a
Sigmoid, despite its label in Figure 4, cannot regress raw false-negative
counts greater than one. PT's gradient-norm-10 clipping is not applied because
ADA-FNP does not specify it.

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

`PTVOCMetric` avoids MMDetection's legacy inclusive `+1` arithmetic. It also
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
probability; detection count; and the final product. Stored scores are
recomputed and checked before selection. Empty-detection images have final
score zero.

The run records its resolved config and fingerprint, atomic stage state,
content-addressed artifacts, checkpoints, selections, and evaluation metrics.
Resume validates the config fingerprint and prior checkpoint/artifact hashes.
Detector resume additionally compares model keys and tensor shapes exactly,
requires optimizer and parameter-scheduler sections, and accepts only the
expected global iteration before handing the checkpoint to MMEngine. Inference
loads model state strictly. Randomness uses the configured seed with
deterministic mode enabled and cuDNN benchmarking disabled.

## Current validation boundary

The concrete execution adapter now connects detector training, FNPM training,
pool scoring, selection, annotation reveal, and teacher evaluation to
MMEngine/MMDetection. Unit tests can validate its stage configuration and
state transitions through injected runtime doubles.

The current workstation interpreter is CPU-only and does not provide the
pinned MMCV/MMEngine runtime. Consequently, a real MMDetection model build,
CUDA NMS/RoIAlign, an end-to-end 40k training run, and the paper's AP50 values
have not been validated here. The implementation must not be described as a
scientific reproduction until these gates are closed. At minimum, report
source-only, 0 percent, 1 percent, and 5 percent C to F results plus component
ablations over three seeds before making that claim.
