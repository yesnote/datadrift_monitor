# ADAOD

ADAOD is a structured implementation workspace for active domain adaptation
methods for object detection. The first supported experiment is ADA-FNP on
Cityscapes to Foggy Cityscapes (`beta=0.02`) with a PT-compatible, BN-free
VGG16 Faster R-CNN detector implemented on MMDetection 3.3.0.

## Supported reproduction target

- Source: Cityscapes train, 2,975 images
- Target pool: Foggy Cityscapes train, 2,975 images
- Evaluation: Foggy Cityscapes val, 500 images
- Categories: the eight PT instance categories in PT registry order
- Budgets: 1 percent by default (30 images), or 5 percent (149 images)
- Training: 40,000 detector iterations with five acquisitions at 5k--25k

Reference implementations under `code_refs` are read-only inputs. Concrete
method code lives under `methods/<method>`, reusable project behavior lives in
`methods/common`, and the vendored `mmdet` tree remains unmodified.

## Prepare the data

Create the following directory junctions. Their targets are local machine
paths and are not stored in configuration:

```text
data/Cityscapes/
|-- gtFine
|-- leftImg8bit
`-- leftImg8bit_foggy
```

Then validate the complete layout and build the annotation cache:

```powershell
python tools/internal/prepare_cityscapes.py
```

Generated source, annotation-free target-pool, target-oracle, validation, and
cache-manifest JSON files are written below
`work_dirs/.dataset_cache/cityscapes-to-foggy`.

## Prepare the pretrained VGG16 asset

A normal run downloads and checksum-verifies the Caffe-converted VGG16 asset
when it is absent. It can also be prepared explicitly:

```powershell
python tools/internal/prepare_pretrained.py `
  --url https://zenodo.org/records/4515252/files/vgg16_caffe.pth?download=1 `
  --sha256 736b4bd0b787438253ea1926f9a02730b2eedbf0e48df243457d17133fe8850e `
  --output work_dirs/pretrained/vgg16_caffe.pth
```

The loader verifies the SHA256 before using the checkpoint and imports only
the VGG convolution tensors. A missing or corrupt asset is an explicit error;
`--offline` disables downloading.

## Inspect or run ADA-FNP

```powershell
python -m tools.run_adaod --list-methods
python -m tools.run_adaod --method ada-fnp --dry-run
python -m tools.run_adaod --method ada-fnp --budget-percent 1 --seed 0
```

Resume the same resolved experiment with:

```powershell
python -m tools.run_adaod --method ada-fnp --budget-percent 1 --seed 0 --resume
```

Use `--run-directory` to choose another repository-relative run directory and
`--offline` to require all external assets to be present locally. Before a
model run, install the pinned CUDA stack from `requirements/README.md` and run
`python tools/check_environment.py`.

The execution backend is implemented, but an end-to-end MMDetection/CUDA run
and reproduction of the paper's AP50 numbers have not been completed in the
current CPU-only environment. See `docs/reproducibility.md` for the exact
boundary between implementation parity and scientific reproduction.
