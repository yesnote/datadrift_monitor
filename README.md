# ALOD

ALOD is a local-source implementation workspace for active learning in object
detection. It is organized around one user choice:

```text
method + detector + dataset
```

The runner resolves the paper reproduction settings from that choice and uses
the local source tree directly. This repository is not installed with
`pip install -e .` or `setup.py install`.

## Supported Targets

Current implemented target:

| Method | Detector | Dataset | Notes |
| --- | --- | --- | --- |
| PPAL | RetinaNet | PASCAL VOC | DCUS + CCMS reproduction path |
| PAL | RetinaNet | PASCAL VOC | LIUS + GUIDE reproduction path |
| Random | RetinaNet | PASCAL VOC | Baseline acquisition |
| Entropy | RetinaNet | PASCAL VOC | Baseline acquisition |

`code_refs/` contains archived upstream/reference implementations only. Runtime
code is copied into this repository and should be imported from the local ALOD
source tree, not from `code_refs/`.

## Repository Layout

- `tools/run_active_learning.py`: main runner for executing AL rounds.
- `configs/catalog/`: supported method/detector/dataset presets.
- `configs/alod_mmdet/`: minimal MMDetection configs used by ALOD runs.
- `methods/ppal/`: PPAL method implementation.
- `methods/pal/`: PAL method implementation.
- `mmdet/`: local MMDetection backend used by train/test entrypoints.
- `datasets/`: dataset and active learning pool preparation utilities.
- `docs/`: implementation notes, plans, and run logs.

## Prepare Data

Activate the environment:

```powershell
conda activate alod
```

Install the framework stack first. The currently validated target is
PyTorch 1.10, torchvision 0.11, and `mmcv-full` 1.3.17-1.5.0; install wheels
that match the local CUDA/driver setup. Then install ALOD's pip-level runtime:

```powershell
pip install -r requirements.txt
```

Prepare VOC active learning pools after `data/VOCdevkit` contains `VOC2007`
and `VOC2012`:

```powershell
python -B datasets/prepare_voc_active_learning.py --vocdevkit data/VOCdevkit --n-labeled 827 --n-diff 1 --seed 0
```

Prepare the RetinaNet ResNet-50 backbone checkpoint:

```powershell
python -B tools/prepare_pretrain_models.py --output-dir data/pretrain_models
```

PAL GUIDE uses an image embedding cache:

```powershell
pip install -r requirements/pal_embeddings.txt
python -B tools/build_pal_vit_embeddings.py --ann-file data/active_learning/voc/voc_827_labeled_1.json --ann-file data/active_learning/voc/voc_827_unlabeled_1.json --image-root data/VOCdevkit --output work_dirs/pal_embeddings/voc_google_vit_embeddings.npy --device auto
```

## Run

List supported presets:

```powershell
python -B tools/run_active_learning.py --list-presets
```

Run one PAL round:

```powershell
python -B tools/run_active_learning.py --method pal --detector retinanet --dataset voc --rounds 1 --gpus 1
```

Useful method aliases:

- `ppal`: PPAL DCUS + CCMS.
- `pal`, `pal:guide`, `pal/full`: PAL LIUS + GUIDE.
- `pal:lius`: PAL LIUS only.
- `random`: random acquisition baseline.
- `entropy`: entropy acquisition baseline.
