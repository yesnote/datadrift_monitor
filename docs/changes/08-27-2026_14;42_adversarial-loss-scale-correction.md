# Adversarial loss scale correction

## Problem

The shared domain discriminator loss averaged the separately reduced source
and target binary cross-entropy losses with an additional factor of `0.5`.
This made the configured adversarial weight of `0.01` behave like `0.005` in
both AADA and ADA-FNP.

The AADA objective and ADA-FNP supplementary formulation add the source-domain
and target-domain expectations without this extra factor.

## Change

`domain_discriminator_loss` now returns the sum of the independently averaged
source and target losses. When labeled and unlabeled target branches are both
present, `compute_multi_target_domain_loss` continues to average the target
branches, producing:

```text
L_source + mean(L_target_labeled, L_target_unlabeled)
```

The configured adversarial loss weight remains `0.01`. Existing checkpoints
were trained with the old scale and must not be resumed for reproduction runs.

## Validation

- Parsed the modified Python module with `ast.parse` without writing bytecode.
- Evaluated the loss functions with fixed logits and confirmed source and
  target terms are summed while multiple target branches are averaged.
- Checked the modified files with `git diff --check`.
- No training, smoke, test, cache, or experiment artifacts were created.
