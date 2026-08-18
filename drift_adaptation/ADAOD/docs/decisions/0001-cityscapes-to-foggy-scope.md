# Cityscapes to Foggy Cityscapes scope

The first ADAOD scenario is intentionally limited to the ADA-FNP paper's C to
F setting.

- Source training: Cityscapes train, 2,975 images
- Target pool: Foggy Cityscapes train, beta 0.02, 2,975 images
- Evaluation: Foggy Cityscapes val, beta 0.02, 500 images
- Detector: Faster R-CNN with a BN-free VGG16
- Internal class order: person, rider, car, truck, bus, train, motorcycle,
  bicycle

Cityscapes val is excluded from source training because it has the same scenes
and geometry as Foggy Cityscapes val. Foggy beta 0.005 and 0.01 files are not
part of this scenario.

Annotations are converted from gtFine polygon JSON. Deleted objects and
non-instance classes are excluded, group objects are marked as crowd, bounding
boxes use COCO xywh externally and half-open xyxy internally, and empty images
remain in the pool. AP50 uses continuous-area VOC interpolation to match the
PT registration of the dataset as VOC 2012.

One percent and five percent budgets are rounded to the nearest image with
half values rounded upward, producing 30 and 149 images. Remainders are
assigned to earlier acquisition rounds.
