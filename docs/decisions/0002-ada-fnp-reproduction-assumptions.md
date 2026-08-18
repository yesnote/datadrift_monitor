# ADA-FNP reproduction assumptions

- False negatives use score-ordered, class-aware, one-to-one matching at IoU
  0.5 over the post-NMS top 100 detections. There is no extra confidence cut.
- A sample with no predicted boxes has zero localization, entropy, and final
  acquisition score.
- Classification entropy excludes background without renormalizing foreground
  probabilities.
- Constant score components normalize to 0.5. Domain probabilities are
  clamped with epsilon 1e-6; non-finite scores are errors.
- MC Dropout is placed after fc6 and fc7 with probability 0.1. The RPN runs
  once and ten RoI passes reuse fixed proposals. Means are computed before one
  NMS operation.
- FNPM predicts a non-negative raw false-negative count with Softplus and is
  optimized for 2,000 iterations per round at learning rate 1e-4.
- The paper EMA decay 0.9996 maps to MMDetection teacher hook momentum 0.0004.
- Unlabeled target losses contain RPN and RoI classification terms only.
- Pseudo labels use localization variance at most 0.1 and no added hard
  confidence threshold.
