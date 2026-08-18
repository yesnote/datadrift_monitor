# Architecture

ADAOD keeps method code outside the vendored detection framework. Concrete
method code, configuration, documentation, and tests belong to one subtree
under `methods`. Reusable behavior is promoted to `methods/common` only after
its inputs, outputs, mathematical meaning, and lifecycle are shared.

The dependency direction is:

```text
tools -> configs/catalog -> methods/<method> -> methods/common -> mmdet
```

The `mmdet` package does not import project modules. Project extensions use
MMDetection registries through `custom_imports`. Reference code under
`code_refs` is never imported at runtime.

`tools` contains command-line and launch glue only. Training losses,
acquisition scores, and method-specific dispatch stay in method packages.
The common runner executes a serial list of stages and has no method-name
branches.

Dataset input under `data` is read-only. Generated annotations, checkpoints,
scores, selections, and state belong under `work_dirs`.
