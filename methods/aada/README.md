# AADA under the ADA-FNP protocol

This package implements Active Adversarial Domain Adaptation as the active
sampling baseline in the ADA-FNP Cityscapes-to-Foggy comparison. It does not
reproduce AADA's original classification experiment protocol. Dataset,
detector, 40k schedule, five acquisition milestones, percentage budget, and
AP50 evaluator are identical to ADA-FNP.

## Method boundary

`AadaDetector` trains full Faster R-CNN detection losses on Cityscapes source
and cumulatively revealed Foggy target images. The Progressive-DA image-level
discriminator receives source as domain 1 and both labeled and unlabeled
target images as domain 0 through gradient reversal. There is no teacher EMA,
false-negative predictor, pseudo-label loss, MC Dropout, or strong/weak
teacher-student branch.

Training uses the shared 600-pixel resize and random horizontal flip but does
not apply ADA-FNP's teacher/student photometric strong augmentation.

Pool acquisition uses one deterministic detector pass. For each image it
computes the mean foreground-class entropy over post-NMS detections and the
domain importance weight `(1-D)/D`, where `D` is the spatially averaged source
probability. Their raw product is the AADA score. An image with no detections
has score zero. The score artifact retains both components, `D`, detection
count, and final score.

## Shared protocol

- Cityscapes train to Foggy Cityscapes beta 0.02 train/val
- BN-free VGG16 Faster R-CNN, short edge 600, Caffe BGR normalization
- SGD 0.02, momentum 0.9, weight decay 0.0001, warm-up 0--400
- LR drops at 30k and 35k; training ends at 40k
- source, labeled-target, and unlabeled-target batch size 4 when present
- active rounds at 5k, 10k, 15k, 20k, and 25k
- the total percentage budget is divided across all five rounds
- checkpoint AP50 at 5k, 10k, 15k, 20k, 25k, and 40k

For 2,975 target images, 1 percent resolves to 30 images and selects six per
round. Five percent resolves to 149 images and selects 30, 30, 30, 30, and 29.
The ADA-FNP paper reports AADA AP50 37.2 at 1 percent and 38.5 at 5 percent.

```powershell
python -B -m tools.run_adaod `
  --method aada `
  --dataset cityscapes-to-foggy `
  --detector faster-rcnn-vgg16 `
  --budget-percent 1 `
  --seed 0
```

Outputs use the common timestamped path below
`work_dirs/runs/aada/cityscapes-to-foggy/faster-rcnn-vgg16/seed-<seed>`.
