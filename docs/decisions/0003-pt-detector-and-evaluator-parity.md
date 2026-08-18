# PT detector and evaluator parity

ADAOD keeps MMDetection as the execution framework while reproducing PT's
observable Faster R-CNN VGG16 and Pascal-VOC behavior through explicit
configuration and project extensions. Framework defaults are not considered a
valid substitute for these values.

Detector decisions:

- load only convolution tensors from the pinned Caffe VGG16 asset;
- use Caffe BGR normalization, unit standard deviation, and no RGB swap;
- freeze the first two VGG stages and omit pool5;
- use anchor sizes 128, 256, and 512 at stride 16 with center offset 0.0;
- retain each GT's best RPN anchor with `min_pos_iou=0.0`, use positive
  fraction 0.25, and use PT's train/test proposal limits;
- use aligned 7-by-7 RoIAlign with adaptive sampling;
- use two 1,024-wide fully connected layers with Detectron2 C2 Xavier
  initialization and class-specific bbox regression; and
- use PT's 400-iteration linear warm-up and 30k/35k LR milestones.

Evaluation decisions:

- preserve zero-based, half-open boxes throughout conversion and evaluation;
- quantize predictions as Detectron2's Pascal text writer does;
- use strict `IoU > 0.5`, not `>= 0.5`;
- disable legacy `+1` coordinate arithmetic; and
- use VOC2012 continuous-area AP50 on the percentage scale.

The custom `PTVOCMetric` exists because MMDetection's stock VOC metric forces
legacy inclusive-coordinate arithmetic, which would change the result for the
PT-compatible cache. These choices target implementation parity; measured
paper-level parity remains contingent on the full GPU experiments.

MMDetection is still a framework port rather than a bit-identical execution of
PT. Its OpenCV-backed resize path and random-number consumption can differ
from Detectron2/PIL at the pixel and seed-trajectory level even when the
configured sizes and augmentation distributions match. GPU parity runs must
measure the effect instead of treating configuration parity as numerical
identity.
