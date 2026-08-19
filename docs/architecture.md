# Architecture

ADAOD keeps concrete method behavior outside the vendored detection framework.
The dependency direction is:

```text
tools -> method manifest -> methods/<method> -> methods/common -> mmdet
```

`configs/catalog` contains method-neutral dataset, detector, and runtime
metadata. A method manifest supplies its config factory, serial plan factory,
and explicit `executor_module`. `tools/run_adaod.py` discovers that manifest,
resolves a deterministic configuration, imports the executor module, and gives
the resulting stage list and executor registry to the common runner. The
runner knows executor keys and artifacts but contains no method-name branch.

Concrete ADA-FNP models, training logic, acquisition logic, stage definitions,
and execution adapters live under `methods/ada_fnp`. Behavior is placed in
`methods/common` only when its input/output contract and lifecycle are shared
by more than one method or are inherently method-neutral. The `mmdet` package
does not import project modules; project models, transforms, and metrics enter
MMDetection through config `custom_imports` and explicit registry names.
`code_refs` is never a runtime dependency.

The shared layer is intentionally small:

- `methods/common/artifacts.py` owns atomic bytes/JSON, SHA256, contained
  artifact paths, and `ArtifactStore`.
- `methods/common/external_assets.py` owns checksum-pinned HTTPS assets.
- `methods/common/acquisition/score_artifacts.py` owns the common score and
  selection artifact schema.
- `methods/common/data/cityscapes/layout.py`, `conversion.py`, and `reveal.py`
  separate layout validation, deterministic conversion, and annotation reveal.
- `methods/common/engine/executor_loader.py` loads only the module named by a
  manifest's `executor_module` field.

ADA-FNP is separated by responsibility rather than lifecycle fragments:

```text
methods/ada_fnp/
|-- schedule.py
|-- probabilistic_teacher_augmentation.py
|-- acquisition/{mc_dropout.py,scoring.py}
|-- models/{detector.py,domain_adaptation.py,
|           false_negative_predictor.py,mc_dropout_roi_head.py}
|-- training/{false_negative_matching.py,false_negative_training.py,
|             pseudo_labeling.py}
`-- execution/{stages.py,mmdet_backend.py,mmdet_config.py,
               mmdet_checkpoints.py,run_files.py}
```

The MMDetection registry names mirror those responsibilities:
`AdaFnpDetector`, `AdaFnpDetectorBranch`, `AdaFnpDomainDiscriminator`,
`AdaFnpMonteCarloDropoutRoIHead`,
`ProbabilisticTeacherStrongAugmentation`, and
`Detectron2PascalVocMetric`.

## Execution flow

```text
resolved config + plan
        |
        v
common StageRunner
        |
        +-- prepare asset and dataset cache
        +-- MMEngine detector segment
        +-- false-negative predictor optimization
        +-- current-pool scoring -> selection -> oracle reveal
        +-- next MMEngine detector segment
        `-- final teacher Detectron2-compatible Pascal-VOC evaluation
```

`methods/ada_fnp/execution/stages.py` is the boundary between method-neutral
stage orchestration and the MMEngine/MMDetection runtime. It registers the
descriptive `ada_fnp.*` executor keys. `mmdet_backend.py` performs model work,
`mmdet_config.py` builds stage-specific configs, `mmdet_checkpoints.py` owns
strict detector checkpoint handling, and `run_files.py` owns run-local paths
and pool manifests. Detector segments retain a single global iteration
schedule from `schedule.py`. The first 5k segment has no target labels or
teacher pseudo-label loss; later segments use the latest selected-only labeled
manifest and the committed unlabeled-pool manifest.

Dataset inputs below `data/Cityscapes` are read-only junctions. Stable converted
data belong under `work_dirs/.dataset_cache`; run-local labeled and unlabeled
views, checkpoints, scores, selections, pool transitions, resolved config, and
state belong under the selected `work_dirs` run directory. All stage outputs
are written atomically and referenced by content hashes where the artifact
contract requires it.

The current manifest and run-state schemas are version 2. Earlier state files
and the former abbreviated registry/executor names are intentionally not
translated. Restart those experiments in a fresh run directory.
