# Cityscapes to Foggy Cityscapes scope

The first ADAOD scenario is intentionally limited to the ADA-FNP paper's C to
F setting.

- Source training: Cityscapes train, 2,975 images
- Target pool: Foggy Cityscapes train, beta 0.02, 2,975 images
- Evaluation: Foggy Cityscapes val, beta 0.02, 500 images
- Detector: Faster R-CNN with a BN-free VGG16
- Internal PT class order: `truck`, `car`, `rider`, `person`, `train`,
  `motorcycle`, `bicycle`, `bus`

Cityscapes val is excluded from source training because it contains the same
scenes and geometry as Foggy Cityscapes val. Foggy beta 0.005 and 0.01 files
are outside this scenario.

Annotations are converted from `gtFine` polygon JSON using exact matching for
the eight PT labels. Deleted objects, non-instance classes, and `*group`
regions are excluded; group regions are not added as crowd annotations. The
converter reproduces PT's VOC serialization followed by Detectron2's loader,
yielding zero-based lower bounds and half-open upper bounds. Empty images
remain in the pool.

Cityscapes handling is split by responsibility under
`methods/common/data/cityscapes`: `layout.py` validates the junctions and
declares dataset constants, `conversion.py` prepares the deterministic cache,
and `reveal.py` creates selected-only target annotations from the read-only
oracle.

The required local layout is `data/Cityscapes/{gtFine,leftImg8bit,
leftImg8bit_foggy}`. These entries are read-only junctions; generated
annotations are stored only below `work_dirs`.

One percent and five percent budgets are rounded to the nearest image with
half values rounded upward, producing 30 and 149 images. Remainders are
assigned to earlier acquisition rounds, so the five-round allocations are
`6,6,6,6,6` and `30,30,30,30,29`.
