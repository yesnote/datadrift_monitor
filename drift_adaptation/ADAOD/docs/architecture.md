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

Concrete model, training, acquisition, stage, and execution code lives under
`methods/<method>`. Behavior is placed in
`methods/common` only when its input/output contract and lifecycle are shared
by more than one method or are inherently method-neutral. The `mmdet` package
does not import project modules; project models, transforms, and metrics enter
MMDetection through config `custom_imports` and explicit registry names.
`code_refs` is never a runtime dependency.

The shared layer is intentionally small:

- `methods/common/artifacts.py` owns atomic bytes/JSON, SHA256, contained
  artifact paths, and `ArtifactStore`.
- `methods/common/external_assets.py` owns checksum-pinned HTTPS assets.
- `methods/common/progress.py` owns the process-wide terminal progress line;
  long-running stages receive the same reporter instead of creating nested
  progress displays.
- `methods/common/acquisition/score_artifacts.py` owns the common score and
  selection artifact schema.
- `methods/common/acquisition/budget.py` and `domain_uncertainty.py` own the
  percentage budget and entropy/domain-diversity primitives shared by
  ADA-FNP and AADA.
- `methods/common/data/cityscapes/layout.py`, `conversion.py`, and `reveal.py`
  separate layout validation, deterministic conversion, and annotation reveal.
- `methods/common/engine/executor_loader.py` loads only the module named by a
  manifest's `executor_module` field.
- `methods/common/protocols` owns the ADA-FNP comparison schedule and generic
  active-detection plan.
- `methods/common/execution` owns C-to-F preparation, pool paths, strict
  detector continuation, selection, reveal, and checkpoint evaluation.
- `methods/common/mmdet` owns shared configuration plumbing, runtime, the
  Progressive-DA discriminator, and deterministic probability RoI head.

ADA-FNP is separated by responsibility rather than lifecycle fragments:

```text
methods/ada_fnp/
|-- schedule.py
|-- probabilistic_teacher_augmentation.py
|-- acquisition/{mc_dropout.py,scoring.py}
|-- models/{detector.py,
|           false_negative_predictor.py,mc_dropout_roi_head.py}
|-- training/{false_negative_matching.py,false_negative_training.py,
|             pseudo_labeling.py}
`-- execution/{stages.py,mmdet_backend.py,mmdet_config.py}
```

AADA keeps only the behavior that differs from ADA-FNP:

```text
methods/aada/
|-- acquisition/scoring.py
|-- models/detector.py
|-- configs/{default.py,cityscapes_to_foggy.py}
`-- execution/{stages.py,mmdet_backend.py,mmdet_config.py}
```

The MMDetection registry names mirror those responsibilities:
`AdaFnpDetector`, `AdaFnpDetectorBranch`, `AadaDetector`,
`ProgressiveDomainDiscriminator`,
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

`tools/run_adaod.py` creates one runtime-only progress reporter and passes it
through `ExecutionContext`. `StageRunner`, MMEngine hooks, acquisition,
dataset preparation, and verified asset downloads reset and update that same
line. `methods/common/mmdet/progress.py` also separates the MMEngine console
handler from its file handler: repetitive INFO output is hidden from the
terminal while the timestamped file remains at INFO level.

Each method's `execution/stages.py` binds its scorer and model work to common
active-detection stages. Detector segments retain one global 40k schedule from
`methods/common/protocols`. The first 5k segment has no target labels; later
segments use the latest selected-only labeled manifest and committed
unlabeled-pool manifest.

Dataset inputs below `data/Cityscapes` are read-only junctions. Stable converted
data belong under `work_dirs/.dataset_cache`; run-local labeled and unlabeled
views, checkpoints, scores, selections, pool transitions, resolved config, and
state belong under the selected `work_dirs` run directory. All stage outputs
are written atomically and referenced by content hashes where the artifact
contract requires it.

The current manifest and run-state schemas are version 2. Earlier state files
and the former abbreviated registry/executor names are intentionally not
translated. Restart those experiments in a fresh run directory.
