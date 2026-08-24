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
`methods/common`, and ADAOD-specific modules stay outside the vendored `mmdet`
tree. The one framework compatibility patch is documented in
`docs/upstream.md`.

The method manifest identifies its execution entry point with
`executor_module`; the common runner imports that module and calls its
`create_executor_registry` factory. ADA-FNP exposes
`methods.ada_fnp.execution.stages`, so method-specific stage names and runtime
behavior do not leak into the common engine.

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
the VGG convolution tensors. An existing valid file is reused. A missing file
is downloaded automatically, while a corrupt cached file is an explicit error.

## Inspect or run ADA-FNP

```powershell
python -m tools.run_adaod --list-methods
python -m tools.run_adaod --method ada-fnp --budget-percent 1 --seed 0
```

Interactive runs use one reusable `tqdm` line for all 29 stages. Detector and
false-negative predictor training show only total loss beside the current
progress; pool scoring and evaluation show completed work. Routine
MMEngine configuration and interval output are kept out of the terminal but
remain in the timestamped `.log`, `vis_data/scalars.json`, and resolved run
configuration. A successful run ends with one compact JSON summary.

By default, each execution is stored under
`work_dirs/runs/<method>/<scenario>/<detector>/seed-<seed>/MM-DD-YYYY_HH_mm`
using the local 24-hour clock. Use `--run-directory` to choose an exact
repository-relative run directory instead. Before a model run, install the
pinned CUDA stack from `requirements/README.md` and run
`python tools/check_environment.py`.

The methods-structure refactor introduced manifest API version 2, run-state
schema version 2, and descriptive MMDetection registry names. Runs created by
the earlier schema or registry names are intentionally incompatible. Start
them again in a fresh run directory. ADAOD does not resume an interrupted
user run or overwrite a nonempty run directory.

Completion of the full 40k experiment and parity with the paper's AP50 values
remain unvalidated. See `docs/reproducibility.md` for the exact boundary.
