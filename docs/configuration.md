# Configuration

Root catalogs contain method-neutral dataset, detector, and runtime metadata.
A concrete method owns its defaults and experiment preset under its own
package. The selected manifest is discovered from `methods/*/manifest.py`; no
second method list is maintained.

Resolution order is method defaults, dataset catalog, detector catalog,
runtime catalog, then explicit command-line overrides. The resolved mapping is
serialized deterministically and assigned a SHA256 fingerprint. A resume is
rejected if its saved fingerprint differs from the newly resolved mapping.

The current supported keys are:

| Kind | Key | Meaning |
| --- | --- | --- |
| method | `ada-fnp` | ADA-FNP five-round active adaptation |
| dataset | `cityscapes-to-foggy` | Cityscapes to Foggy, beta 0.02 |
| detector | `faster-rcnn-vgg16` | PT-compatible BN-free VGG16 Faster R-CNN |
| runtime | `default` | deterministic local work directory, no launcher |

Command-line overrides currently exposed by `tools.run_adaod` are acquisition
budget percentage, seed, dataset, detector, runtime, and run directory. A
normal run uses the deterministic path
`work_dirs/runs/<method>/<scenario>/<detector>/seed-<seed>`.

All configured repository assets and dataset inputs use repository-relative
paths. Workstation-specific dataset locations appear only as the targets of
the junctions below `data/Cityscapes`. During a run, the execution adapter
resolves cache paths and replaces the generic target-labeled and
target-unlabeled annotation paths with run-local manifests for the currently
committed pool.

`--offline` forbids pretrained downloads. `--resume` requires the existing
resolved config and state under the same run directory. Starting without
`--resume` refuses to overwrite an existing run. Detector continuation fails
before MMEngine resume unless the checkpoint has exact model keys and tensor
shapes, optimizer and parameter-scheduler state, and the expected global
iteration metadata.
