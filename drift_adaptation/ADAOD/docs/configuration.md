# Configuration

Root catalogs contain method-neutral dataset, detector, and runtime metadata.
A concrete method owns its defaults and experiment preset under its own
package. The selected manifest is discovered from `methods/*/manifest.py`; no
second method list is maintained. Each manifest declares an
`executor_module`; for ADA-FNP it is `methods.ada_fnp.execution.stages`, whose
`create_executor_registry` function supplies the method's stage executors.

Resolution order is method defaults, dataset catalog, detector catalog,
runtime catalog, then explicit command-line overrides. The resolved mapping is
serialized deterministically and assigned a SHA256 fingerprint.
`execution/mmdet_config.py` projects the resolved detector, optimizer,
schedule, batch-size, domain-adaptation, teacher-EMA, MC Dropout, and
pseudo-label settings onto every MMDetection stage config. The generated
`resolved_config.json` is therefore the runtime source of truth rather than a
descriptive copy of separate defaults.

ADA-FNP uses an acquisition inference batch size of 4 and a final evaluation
batch size of 4. These values live under the resolved `inference` mapping;
they are independent of the per-domain training batch sizes and are included
in the run fingerprint.

The resolved `pseudo_label` mapping fixes Supplementary indicator thresholds:
`localization_variance_threshold=0.1` in RoI bbox-delta space and
`confidence_threshold=0.5` over the mean foreground class probability. The
detector's separate 0.05 score cutoff only removes low-score candidates before
NMS and is not the pseudo-label confidence threshold.

The current supported keys are:

| Kind | Key | Meaning |
| --- | --- | --- |
| method | `ada-fnp` | ADA-FNP five-round active adaptation |
| dataset | `cityscapes-to-foggy` | Cityscapes to Foggy, beta 0.02 |
| detector | `faster-rcnn-vgg16` | PT-compatible BN-free VGG16 Faster R-CNN |
| runtime | `default` | deterministic local work directory, no launcher |

Command-line overrides currently exposed by `tools.run_adaod` are acquisition
budget percentage, seed, dataset, detector, runtime, and run directory. A
normal run uses the timestamped path
`work_dirs/runs/<method>/<scenario>/<detector>/seed-<seed>/MM-DD-YYYY_HH_mm`,
where the final component is the local start time on a 24-hour clock. An
explicit `--run-directory` remains an exact override and does not receive an
additional timestamp.

All configured repository assets and dataset inputs use repository-relative
paths. Workstation-specific dataset locations appear only as the targets of
the junctions below `data/Cityscapes`. During a run, the execution modules
resolve cache paths and replace the generic target-labeled and
target-unlabeled annotation paths with run-local manifests for the currently
committed pool.

The pinned pretrained checkpoint is reused when present and downloaded
automatically when missing. A cached file with the wrong SHA-256 is rejected.
ADAOD refuses to start in a nonempty run directory; an interrupted experiment
must be started again in a new or explicitly cleared directory. The internal
5k detector segments still form one continuous 40k optimization schedule. A
segment continuation fails before MMEngine loads the preceding checkpoint
unless it has exact model keys and tensor shapes, optimizer and
parameter-scheduler state, and the expected global iteration metadata.

The refactored manifest API and run state are version 2. Configuration uses
the full `false_negative_predictor`, `domain_adaptation`, and descriptive
MMDetection registry names (`AdaFnpDetector`, `AdaFnpDetectorBranch`,
`AdaFnpDomainDiscriminator`, `AdaFnpMonteCarloDropoutRoIHead`,
`ProbabilisticTeacherStrongAugmentation`, and
`Detectron2PascalVocMetric`). The former abbreviated keys and schema-1 run
state are not migrated. Use a fresh `--run-directory` and rerun from the
beginning when an old run exists.
