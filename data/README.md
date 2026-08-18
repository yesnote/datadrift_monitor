# Dataset links

This directory contains local junctions to datasets stored outside the
repository. Dataset files are read-only inputs and must not be committed.

The Cityscapes-to-Foggy Cityscapes scenario expects these junctions:

| Local path | External target |
| --- | --- |
| `data/Cityscapes/gtFine` | Cityscapes `gtFine` directory |
| `data/Cityscapes/leftImg8bit` | Cityscapes `leftImg8bit` directory |
| `data/Cityscapes/leftImg8bit_foggy` | Foggy Cityscapes image directory |

The experiment configuration uses repository-relative local paths only. The
workstation-specific junction targets are intentionally not stored in source
configuration.
