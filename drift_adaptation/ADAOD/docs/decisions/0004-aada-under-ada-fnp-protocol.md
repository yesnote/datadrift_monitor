# Decision 0004: Evaluate AADA under the ADA-FNP protocol

## Decision

AADA is implemented as an algorithmic baseline inside the ADA-FNP C-to-F
experiment protocol. The implementation deliberately does not use the
dataset, detector, fixed per-round budget, or round count from AADA's original
classification experiments.

The shared contract fixes Foggy beta 0.02, VGG16 Faster R-CNN, 40k continuous
optimization, acquisitions at 5k through 25k, the total `budget_percent`
divided across five rounds, and 8-class mean AP50 evaluation. AADA changes only
training and sampling behavior: supervised source/revealed-target detection,
source/target adversarial alignment, and entropy times domain diversity
selection.

AADA training retains the common resize and random flip but omits ADA-FNP's
photometric strong augmentation because that transform belongs to the
teacher/student pseudo-label path rather than the AADA algorithm.

Percentage budget resolution, pool transitions, C-to-F conversion, detector
schedule, checkpoint continuation, selection, reveal, and evaluation have one
implementation under `methods/common`. AADA does not import ADA-FNP modules.

The domain discriminator treats all target images as target-domain samples,
including revealed target images, while their annotations additionally supply
supervised detection loss. The ADA-FNP paper does not fully specify every
internal choice of its AADA reimplementation, so this physical-domain routing
is recorded explicitly rather than hidden as a default.
