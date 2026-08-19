# Remove test and generated artifacts

## Summary

Removed project-owned test source and local validation output that is not
required to run ADAOD. Runtime implementation, configuration, datasets,
pretrained weights, and real experiment state were preserved.

## Removed

- `methods/ada_fnp/tests/`
- `methods/common/tests/`
- `methods/common/mmdet/tests/`
- `tools/tests/`
- `requirements/development.txt`, which contained only pytest dependencies
- `work_dirs/.smoke/`
- generated `.pytest_cache/`, `__pycache__/`, `.pyc`, `.pyo`, `.coverage`, and
  `htmlcov/` artifacts in project/runtime source paths

## Preserved

- `work_dirs/runs/`, including the real seed-0 C-to-F experiment state
- `work_dirs/.dataset_cache/`
- `work_dirs/pretrained/`
- `data/Cityscapes/` junctions and their external targets
- `code_refs/` and existing documentation history

## Verification

Only file inventory, path-boundary checks, and Git diff inspection were used.
No test, smoke, training, or inference command was run during this cleanup.
