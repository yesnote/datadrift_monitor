# ADA-FNP reproduction decisions

The ADA-FNP paper leaves several implementation details implicit. The
following decisions are explicit so that later methods and ablations do not
silently change the baseline.

- False negatives use score-ordered, class-aware, one-to-one matching at IoU
  0.5 or greater over the post-NMS top 100 detections. This
  follows the TIDE false-negative boundary and adds no separate confidence
  threshold.
- A sample with no predicted boxes has zero localization and entropy values,
  and its final multiplicative acquisition score is zero. ADAOD does not add a
  neutral-value exception to favor empty predictions.
- Classification entropy sums the eight foreground terms from the detector's
  softmax distribution. Background is excluded, but the foreground
  probabilities are not renormalized to sum to one.
- Constant score components normalize to 0.5. Domain probabilities are
  clamped with epsilon 1e-6; non-finite scores are errors.
- MC Dropout is placed after fc6 and fc7 with probability 0.1. The RPN runs
  once and ten RoI passes reuse the same proposals. Class probabilities and
  class-specific boxes are averaged before a single multiclass NMS; each
  retained detection's variance follows its proposal/class pair.
- `FalseNegativePredictor` predicts a non-negative raw false-negative count
  with Softplus and is optimized for 2,000 iterations per round at learning
  rate 1e-4. Each round receives a fresh optimizer and cosine scheduler while
  retaining predictor weights.
  Source and selected-target loaders drop their incomplete tail so every
  predictor optimization step preserves the configured per-domain batch size
  of four.
  Figure 4 labels the output as Sigmoid, but that bounded output conflicts with
  direct regression of false-negative counts greater than one; Softplus is the
  chosen resolution of that ambiguity.
- The paper EMA decay 0.9996 maps to MMDetection teacher-hook momentum 0.0004.
- Unlabeled target losses contain RPN and RoI classification terms only.
- Pseudo labels use localization variance at most 0.1 and no additional hard
  confidence threshold.
- Geometric resize and flip are sampled once before weak/strong branching.
  Only the strong branch receives PT photometric augmentation.
- Source and selected-target supervision use only the shared weak view. The PT
  trainer supervises both weak and strong labeled/source views, but ADA-FNP
  Figure 2 and Equations 13--14 assign weak-teacher/strong-student routing to
  the unlabeled target data; the implementation follows the paper here.
- Detector optimization follows PT by clipping the global gradient norm at 10.
  The optimizer wrapper also rejects non-finite gradients before updating the
  detector. This setting is required for stable adaptation training in the
  MMDetection port.

These decisions define the official ADA-FNP reproduction config. Alternative
thresholds, no-augmentation training, or special handling of empty detections
must be expressed as separate ablation configs.
