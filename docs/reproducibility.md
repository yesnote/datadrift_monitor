# Reproducibility

The resolved config, stage plan, repository state, upstream MMDetection commit,
dataset fingerprint, package versions, and random seed are recorded before a
run starts. Detector segments preserve global iteration, optimizer, scheduler,
scaler, sampler, student, teacher, discriminator, FNPM, and random states.

The target active pool contains all 2,975 Foggy train images at beta 0.02.
Target annotations exist only in the oracle cache. The unlabeled dataset index
contains image records without annotations, and the oracle reveals records only
after a committed selection. Evaluation annotations are evaluator-only.

Engineering completion and scientific reproduction are separate. A run is a
scientific reproduction only after source-only, 0 percent, 1 percent, and 5
percent C to F results and component ablations are reported over three seeds.
GPU-dependent claims require the pinned Linux or WSL2 environment gate.
