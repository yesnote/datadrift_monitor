# Architecture

ADAOD keeps concrete method behavior outside the vendored detection framework.
The dependency direction is:

```text
tools -> method manifest -> methods/<method> -> methods/common -> mmdet
```

`configs/catalog` contains method-neutral dataset, detector, and runtime
metadata. A method manifest supplies its config factory, serial plan factory,
and execution plugin. `tools/run_adaod.py` discovers that manifest, resolves a
deterministic configuration, and gives the resulting stage list to the common
runner. The runner knows executor keys, checkpoint policies, and artifacts but
contains no method-name branch.

Concrete ADA-FNP models, training logic, acquisition logic, stage definitions,
and execution adapters live under `methods/ada_fnp`. Behavior is placed in
`methods/common` only when its input/output contract and lifecycle are shared
by more than one method or are inherently method-neutral. The `mmdet` package
does not import project modules; project models, transforms, and metrics enter
MMDetection through `custom_imports` and registries. `code_refs` is never a
runtime dependency.

## Execution flow

```text
resolved config + plan
        |
        v
common StageRunner
        |
        +-- prepare asset and dataset cache
        +-- MMEngine detector segment
        +-- FNPM optimization
        +-- current-pool scoring -> selection -> oracle reveal
        +-- next MMEngine detector segment
        `-- final teacher PTVOC evaluation
```

The ADA-FNP execution plugin is the boundary between method-neutral stage
orchestration and the MMEngine/MMDetection runtime. Detector segments retain a
single global iteration schedule. The first 5k segment has no target labels or
teacher pseudo-label loss; later segments use the latest selected-only labeled
manifest and the committed unlabeled-pool manifest.

Dataset inputs below `data/Cityscapes` are read-only junctions. Stable converted
data belong under `work_dirs/.dataset_cache`; run-local labeled and unlabeled
views, checkpoints, scores, selections, pool transitions, resolved config, and
state belong under the selected `work_dirs` run directory. All stage outputs
are written atomically and referenced by content hashes where the artifact
contract requires it.
