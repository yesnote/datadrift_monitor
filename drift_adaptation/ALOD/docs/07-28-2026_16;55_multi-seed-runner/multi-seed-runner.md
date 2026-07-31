# Multi-Seed Runner

## Change

`tools/run_active_learning.py` now stores every run under a timestamped run
directory and supports running one or more seeds from the same command.

```powershell
python -B tools/run_active_learning.py --method pal --detector retinanet --dataset voc --gpus 1 --seeds 0 1 2
```

The same layout is used for a single seed and for multiple seeds:

```text
work_dirs/<experiment_name>/<MM-DD-YYYY_HH;mm>/
  seed_0/
  seed_1/
  seed_2/
  aggregate_summary.json
```

## Seed Scope

The selected seed is passed to the MMDetection training command through
`tools/train.py --seed <seed>`.

The same seed is also used by ALOD acquisition code:

- random sampling
- entropy fallback sampling
- PAL LIUS/GUIDE budget refill
- PPAL CCMS centroid initialization

PPAL DCUS is deterministic for a fixed model and inference artifact, so it does
not need a separate random seed.

## Aggregate Summary

`aggregate_summary.json` is always written, even when only one seed is run. It
collects seed run paths and round-wise mAP/AP50 values from each
`seed_*/round_XX/eval_*.json`, then reports mean and standard deviation.

