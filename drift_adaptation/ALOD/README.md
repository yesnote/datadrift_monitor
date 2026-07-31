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
| ECPAL | RetinaNet | PASCAL VOC | Error-count prediction acquisition |
| Core-set | RetinaNet | PASCAL VOC | Greedy k-center acquisition |
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
- `methods/ecpal/`: ECPAL method implementation.
- `methods/coreset/`: Core-set method implementation.
- `mmdet/`: local MMDetection backend used directly by `tools/train.py` and
  `tools/test.py`; method code consumes saved artifacts instead of importing it.
- `tools/common/`: runner support code for path handling and automatic input preparation.
- `docs/`: implementation notes, plans, and run logs.

## Prepare Environment

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

For PAL GUIDE with Google ViT embeddings, install the optional embedding
dependencies before running PAL full mode:

```powershell
pip install -r requirements/pal_embeddings.txt
```

## Prepare Data

Place the original VOC data under:

```text
data/VOCdevkit/VOC2007
data/VOCdevkit/VOC2012
```

The runner automatically prepares the VOC0712 oracle JSON, initial active
learning pools, RetinaNet ResNet-50 backbone checkpoint, and PAL GUIDE
embedding cache when the selected experiment needs them. VOC source data is not
downloaded automatically.

## Run

List supported presets:

```powershell
python -B tools/run_active_learning.py --list-presets
```

Run one PAL round:

```powershell
python -B tools/run_active_learning.py --method pal --detector retinanet --dataset voc --rounds 1 --gpus 1
```

Run PAL with the three seeds used for paper-style reporting:

```powershell
python -B tools/run_active_learning.py --method pal --detector retinanet --dataset voc --gpus 1 --seeds 0 1 2
```

The default terminal output is concise. It prepares missing inputs, prints the
resolved run and output directory, then shows separate progress bars for the
current round's train, eval, and method inference steps. Acquisition is printed
as a short result line. Detailed command arguments, MMDetection logs, inference
logs, input preparation details, and acquisition details are saved under the run
directory instead of being streamed to the terminal.

Every run is stored under a timestamp directory. Single-seed and multi-seed
runs use the same layout:

```text
work_dirs/<experiment_name>/<MM-DD-YYYY_HH;mm>/
  seed_0/
    active_learning_plan.json
    run_summary.json
    round_00/
    round_01/
  seed_1/
  seed_2/
  aggregate_summary.json
```

Important output files:

- `work_dirs/.../seed_*/active_learning_plan.json`: full command/acquisition plan.
- `work_dirs/.../seed_*/run_summary.json`: resolved method, detector, dataset, rounds,
  budget, output directory, prepared inputs, and round summary paths.
- `work_dirs/.../seed_*/round_XX/round_summary.json`: per-round step status, durations,
  logs, and acquisition outputs.
- `work_dirs/.../seed_*/round_XX/logs/train.log`: training stdout/stderr.
- `work_dirs/.../seed_*/round_XX/logs/eval.log`: evaluation stdout/stderr.
- `work_dirs/.../seed_*/round_XX/logs/*_inference.log`: method inference stdout/stderr.
- `work_dirs/.../seed_*/round_XX/*_diagnostics.json`: method-specific acquisition
  diagnostics.
- `work_dirs/.../seed_*/round_XX/*_candidates.json`: compact candidate rankings and
  final selection flags for method analysis.
- `work_dirs/.../aggregate_summary.json`: seed-level run paths plus round-wise
  mAP/AP50 mean and standard deviation.

Use `--verbose` when debugging to print the full plan and stream subprocess
output:

```powershell
python -B tools/run_active_learning.py --method pal --detector retinanet --dataset voc --rounds 1 --gpus 1 --verbose
```

## View Metrics

Install the optional dashboard dependencies:

```powershell
pip install -r requirements/dashboard.txt
```

Open the read-only local metrics dashboard:

```powershell
python tools/run_metrics_dashboard.py
```

The dashboard scans `work_dirs` by default and lets you compare validation
mAP/AP50 and train loss/lr curves across methods, seeds, rounds, and timestamped
runs from the sidebar.

Useful method aliases:

- `ppal`: PPAL DCUS + CCMS.
- `pal`, `pal:guide`, `pal/full`: PAL LIUS + GUIDE.
- `pal:lius`: PAL LIUS only.
- `ecpal`: ECPAL error-count prediction.
- `coreset`, `core-set`, `kcenter`: Core-set greedy k-center.
- `random`: random acquisition baseline.
- `entropy`: entropy acquisition baseline.
