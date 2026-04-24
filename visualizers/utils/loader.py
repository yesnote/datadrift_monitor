from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class PerImageSample:
    run_root: str
    group: str
    target: str
    map_type: str
    csv_path: str
    vector: np.ndarray
    vector_len: int


def _resolve_path(raw: str, repo_root: Path) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p.resolve()
    return (repo_root / p).resolve()


def _csv_to_vector(csv_path: Path) -> np.ndarray:
    df = pd.read_csv(csv_path)
    required = {"layer_idx", "filter_idx"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV must contain {sorted(required)}: {csv_path}")

    grad_cols = sorted([c for c in df.columns if c.endswith("_grad")])
    value_cols = grad_cols if grad_cols else (["value"] if "value" in df.columns else [])
    if not value_cols:
        return np.zeros((0,), dtype=np.float32)

    df_sorted = df.sort_values(["layer_idx", "filter_idx"], kind="mergesort")
    vec_parts: list[np.ndarray] = []
    for col in value_cols:
        vals = df_sorted[col].astype(np.float32).to_numpy()
        vec_parts.append(vals.reshape(-1))
    if not vec_parts:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(vec_parts, axis=0).astype(np.float32, copy=False)


def collect_per_image_samples(
    run_roots: list[str],
    *,
    map_types: list[str],
    groups: list[str],
    repo_root: Path,
) -> list[PerImageSample]:
    samples: list[PerImageSample] = []
    map_types_l = [str(v).strip().lower() for v in map_types if str(v).strip()]
    groups_l = [str(v).strip().lower() for v in groups if str(v).strip()]

    for raw_root in run_roots:
        run_root = _resolve_path(str(raw_root), repo_root)
        per_image_root = run_root / "ref_maps" / "per_image"
        if not per_image_root.is_dir():
            continue
        for group in groups_l:
            group_dir = per_image_root / group
            if not group_dir.is_dir():
                continue
            for target_dir in sorted([p for p in group_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
                target = target_dir.name
                for map_type in map_types_l:
                    pattern = f"{map_type}_*.csv"
                    for csv_path in sorted(target_dir.glob(pattern)):
                        vec = _csv_to_vector(csv_path)
                        samples.append(
                            PerImageSample(
                                run_root=str(run_root),
                                group=group,
                                target=target,
                                map_type=map_type,
                                csv_path=str(csv_path),
                                vector=vec,
                                vector_len=int(vec.shape[0]),
                            )
                        )
    return samples


def build_key_matrix(samples: list[PerImageSample]) -> tuple[np.ndarray, list[dict]]:
    if not samples:
        return np.zeros((0, 0), dtype=np.float32), []
    max_len = max(int(s.vector.shape[0]) for s in samples)
    n = len(samples)
    x = np.full((n, max_len), np.nan, dtype=np.float32)
    meta: list[dict] = []
    for i, s in enumerate(samples):
        if s.vector.shape[0] > 0:
            x[i, : s.vector.shape[0]] = s.vector
        meta.append(
            {
                "run_root": s.run_root,
                "group": s.group,
                "target": s.target,
                "map_type": s.map_type,
                "csv_path": s.csv_path,
                "vector_len": int(s.vector_len),
            }
        )
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x, meta

