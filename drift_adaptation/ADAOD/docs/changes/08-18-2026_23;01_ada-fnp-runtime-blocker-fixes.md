# ADA-FNP runtime blocker fixes

This change closes two execution blockers found while auditing the production
ADA-FNP Cityscapes-to-Foggy command.

## Implementation

- The detector-stage builder now consumes the MMEngine `_delete_=True` merge
  directive before passing the replacement mapping to `Runner` and
  `DataLoader`. Missing directives remain valid for injected test/runtime
  configs, while any explicit value other than `True` fails immediately.
- The shared MMDetection runtime loader now calls
  `init_default_scope('mmdet')`. Runner-based training and direct registry
  construction used by FNPM, acquisition scoring, evaluation, and resumed
  processes therefore resolve the same MMDetection registries.
- Unit coverage verifies that both initial and adaptation dataloaders no
  longer contain `_delete_`, without mutating their source config.
- Integration coverage starts a separate Python process from an `mmengine`
  scope, loads the runtime, and builds `MultiBranchDataPreprocessor` through
  the resulting `mmdet` scope.

## Validation

- `conda run -n adaod python tools/check_environment.py` passed Python,
  PyTorch, TorchVision, MMEngine, MMCV, repository-local MMDetection, CUDA,
  NMS, and RoIAlign checks on the NVIDIA GeForce RTX 3090.
- The production initial-stage config built its real 4-source plus
  4-target-unlabeled dataloader, completed one deterministic CUDA training
  iteration, and wrote `iter_1.pth`. The runtime dataloader contained no
  `_delete_` key.
- A second, fresh Python process changed from `mmengine` to `mmdet` scope,
  built `ADAFNPDetector`, and loaded that checkpoint with `strict=True`.
- `conda run -n adaod python -m pytest methods/common methods/ada_fnp/tests
  tools/tests -q -p no:cacheprovider` completed with 205 passed and one
  optional real-layout test skipped.

The smoke tests close these two runtime failures. They do not replace the
40,000-iteration experiment or establish scientific AP50 parity.
