# ADA-FNP

This package implements Active Domain Adaptation with False Negative
Prediction for the Cityscapes to Foggy Cityscapes beta 0.02 scenario.

## Reproduction contract

Training starts with source-supervised and source/target adversarial learning
for 5,000 detector iterations. The student detector and domain discriminator
are then copied to the EMA teacher. At detector iterations 5k, 10k, 15k, 20k,
and 25k the workflow trains FNPM for 2,000 iterations, scores the remaining
target pool, selects the round budget, and reveals only those annotations.
Detector optimizer, scheduler, EMA state, and global iteration continue across
segments. The final teacher is evaluated after iteration 40k.

False-negative targets use class-aware one-to-one matching at IoU 0.5 without
an added confidence threshold. Acquisition combines FNPM count, fixed-proposal
MC localization variance, foreground entropy without foreground
renormalization, and the domain ratio. Images with no detections have final
score zero.

## Dry run

```powershell
python -m tools.run_adaod --method ada-fnp --dry-run
python -m tools.run_adaod --method ada-fnp --budget-percent 5 --dry-run
```

Full execution remains gated on the CUDA/MMCV environment check and the
MMDetection integration tests documented under requirements.
