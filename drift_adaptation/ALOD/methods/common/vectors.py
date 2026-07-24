"""Shared vector math helpers."""

from __future__ import annotations

import math

import numpy as np


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """Return an L2-normalized float32 vector, preserving zero vectors."""

    norm = float(np.linalg.norm(vector))
    if norm <= 0.0 or not math.isfinite(norm):
        return vector.astype(np.float32, copy=False)
    return (vector / norm).astype(np.float32, copy=False)
