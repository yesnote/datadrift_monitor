# ADAOD

ADAOD is a structured implementation workspace for active domain adaptation
methods for object detection. The first supported experiment is ADA-FNP on
Cityscapes to Foggy Cityscapes with a BN-free VGG16 Faster R-CNN detector.

## Current scenario

- Source: Cityscapes train (2,975 images)
- Target pool: Foggy Cityscapes train, beta 0.02 (2,975 images)
- Evaluation: Foggy Cityscapes val, beta 0.02 (500 images)
- Detector: Faster R-CNN with a BN-free VGG16 backbone
- Method: ADA-FNP

Reference implementations under code_refs are read-only inputs. Project
methods live under methods and mmdet remains an upstream framework dependency.

## Prepare the C to F cache

Create the three local junctions described in `data/README.md`, then run:

```powershell
python tools/internal/prepare_cityscapes.py
```

The command validates the 2,975/500 split and writes source, unlabeled target,
separate target oracle, and evaluator JSON under
`work_dirs/.dataset_cache/cityscapes-to-foggy`.

Inspect the method plan without starting a GPU job:

```powershell
python -m tools.run_adaod --list-methods
python -m tools.run_adaod --method ada-fnp --dry-run
```

Before training, create the isolated CUDA environment documented in
`requirements/README.md` and run `python tools/check_environment.py`.
