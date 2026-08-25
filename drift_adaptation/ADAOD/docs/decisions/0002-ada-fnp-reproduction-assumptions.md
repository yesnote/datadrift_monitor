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
- The domain discriminator output is the source-domain probability: source is
  1 and target is 0. This follows the main-paper prose and makes the diversity
  ratio `(1-D)/D` increase for target-like samples. Main Equation 3 and
  Supplementary Equation S4 print the opposite binary-cross-entropy terms;
  that internal notation conflict is treated as a paper typo.
- MC Dropout is placed after fc6 and fc7 with probability 0.1. The RPN runs
  once and ten RoI passes reuse the same proposals. Mean foreground
  probability selects one class per proposal. Its class-specific bbox-head
  delta is averaged and decoded once, and its unbiased delta variance follows
  the proposal through one class-aware NMS call. This implements
  Supplementary Equations S6--S8 without image-coordinate normalization.
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
- Pseudo labels implement both Supplementary indicators: mean bbox-delta
  variance at most 0.1 and foreground confidence at least 0.5. The paper does
  not publish the numeric confidence threshold; 0.5 is the fixed reproduction
  decision and is independent of the detector's 0.05 candidate score cutoff.
- Geometric resize and flip are sampled once before weak/strong branching.
  Only the strong branch receives PT photometric augmentation.
- Source and selected-target supervision use PT strong photometric
  augmentation, following Supplementary Equations S2 and S9. Unlabeled target
  data uses the shared-geometry weak-teacher and strong-student views.
- Detector optimization follows PT by clipping the global gradient norm at 10.
  The optimizer wrapper also rejects non-finite gradients before updating the
  detector. This setting is required for stable adaptation training in the
  MMDetection port.

These decisions define the official ADA-FNP reproduction config. Alternative
thresholds, no-augmentation training, or special handling of empty detections
must be expressed as separate ablation configs.
