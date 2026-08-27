# List-valued loss prefixing

## Problem

MMDetection can return RPN losses as lists containing one tensor per feature
level. The shared loss-prefixing helper attempted to multiply the entire list
by a floating-point weight, so AADA stopped on its first detector iteration.

## Change

`methods/common/mmdet/losses.py` now applies the configured weight to each
element of list- or tuple-valued losses while retaining the original container
type. Scalar tensor losses keep the existing behavior.

This common fix applies to both AADA and ADA-FNP without changing loss names,
weights, tensor devices, or gradient flow.

## Validation

- Parsed the modified Python module with `ast.parse` without writing bytecode.
- Checked the modified files with `git diff --check`.
- No training, smoke, or test artifacts were created.
