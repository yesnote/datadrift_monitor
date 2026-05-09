from __future__ import annotations

from dataclasses import dataclass
import json
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


PREDICTION_DUMP_REQUIRED_COLUMNS = {
    "image_id",
    "image_path",
    "pred_idx",
    "raw_pred_idx",
    "xmin",
    "ymin",
    "xmax",
    "ymax",
    "cx",
    "cy",
    "w",
    "h",
    "area",
    "aspect_ratio",
    "anchor_cx",
    "anchor_cy",
    "anchor_w",
    "anchor_h",
    "anchor_area",
    "anchor_aspect_ratio",
    "bbox_anchor_dx",
    "bbox_anchor_dy",
    "bbox_anchor_center_l2",
    "bbox_anchor_w_ratio",
    "bbox_anchor_h_ratio",
    "bbox_anchor_area_ratio",
    "bbox_anchor_log_w_ratio",
    "bbox_anchor_log_h_ratio",
    "bbox_anchor_log_area_ratio",
    "bbox_anchor_aspect_ratio_diff",
    "obj",
    "cls_conf",
    "score",
    "pred_class",
    "pred_class_id",
    "null_obj",
    "null_cls_conf",
    "null_score",
    "obj_null_abs_diff",
    "cls_null_abs_diff",
    "score_null_abs_diff",
    "cls_entropy",
    "cls_entropy_norm",
    "cls_uniform_kl",
    "max_iou",
    "tp",
}


def _read_prediction_dump_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = sorted(PREDICTION_DUMP_REQUIRED_COLUMNS.difference(df.columns))
    if missing:
        raise ValueError(f"prediction_dump.csv is missing required columns {missing}: {csv_path}")
    df = df.copy()
    df["source_csv"] = str(csv_path)
    df["source_run"] = str(csv_path.parent)
    numeric_cols = [
        "image_id",
        "pred_idx",
        "raw_pred_idx",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "cx",
        "cy",
        "w",
        "h",
        "area",
        "aspect_ratio",
        "anchor_cx",
        "anchor_cy",
        "anchor_w",
        "anchor_h",
        "anchor_area",
        "anchor_aspect_ratio",
        "bbox_anchor_dx",
        "bbox_anchor_dy",
        "bbox_anchor_center_l2",
        "bbox_anchor_w_ratio",
        "bbox_anchor_h_ratio",
        "bbox_anchor_area_ratio",
        "bbox_anchor_log_w_ratio",
        "bbox_anchor_log_h_ratio",
        "bbox_anchor_log_area_ratio",
        "bbox_anchor_aspect_ratio_diff",
        "obj",
        "cls_conf",
        "score",
        "pred_class_id",
        "null_obj",
        "null_cls_conf",
        "null_score",
        "obj_null_abs_diff",
        "cls_null_abs_diff",
        "score_null_abs_diff",
        "cls_entropy",
        "cls_entropy_norm",
        "cls_uniform_kl",
        "max_iou",
        "tp",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.replace([np.inf, -np.inf], np.nan)


def collect_prediction_dump_data(
    csv_paths: list[str],
    run_roots: list[str],
    *,
    repo_root: Path,
    max_rows: int = 0,
) -> tuple[pd.DataFrame, list[dict]]:
    paths: list[Path] = []
    for raw in csv_paths:
        csv_path = _resolve_path(str(raw), repo_root)
        paths.append(csv_path)
    for raw in run_roots:
        run_root = _resolve_path(str(raw), repo_root)
        paths.append(run_root / "prediction_dump.csv")

    unique_paths = []
    seen = set()
    for p in paths:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique_paths.append(p.resolve())

    if not unique_paths:
        raise ValueError("input.csv_paths or input.run_roots is required.")

    frames = []
    summaries = []
    for csv_path in unique_paths:
        if not csv_path.is_file():
            raise FileNotFoundError(f"prediction dump CSV not found: {csv_path}")
        df = _read_prediction_dump_csv(csv_path)
        summary_path = csv_path.parent / "prediction_dump_summary.json"
        run_summary = {}
        if summary_path.is_file():
            with open(summary_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    run_summary = loaded
        frames.append(df)
        summaries.append(
            {
                "csv_path": str(csv_path),
                "run_root": str(csv_path.parent),
                "num_rows": int(df.shape[0]),
                "num_images_with_predictions": int(df["image_id"].nunique(dropna=True)) if "image_id" in df.columns else 0,
                "total_images": int(run_summary.get("total_images", 0) or 0),
                "total_predictions": int(run_summary.get("total_predictions", df.shape[0]) or 0),
            }
        )

    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if max_rows > 0 and int(merged.shape[0]) > max_rows:
        merged = merged.sample(n=max_rows, random_state=0).sort_index().reset_index(drop=True)
    return merged, summaries
