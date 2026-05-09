from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from utils.loader import build_key_matrix, collect_per_image_samples
from utils.loader import collect_prediction_dump_data
from utils.plot import save_pca_html, save_prediction_distribution_plots

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent


def _normalize_list(raw, default: list[str]) -> list[str]:
    if raw is None:
        return list(default)
    if isinstance(raw, str):
        t = raw.strip()
        return [t] if t else list(default)
    if isinstance(raw, (list, tuple)):
        vals = [str(v).strip() for v in raw if str(v).strip()]
        return vals if vals else list(default)
    return list(default)


def _limit_samples_per_group(samples: list, max_num: int) -> list:
    if max_num <= 0:
        return samples
    out = []
    seen = {}
    for s in samples:
        g = str(s.group)
        c = int(seen.get(g, 0))
        if c >= max_num:
            continue
        out.append(s)
        seen[g] = c + 1
    return out


def run_visualize(config: dict, run_dir: Path) -> dict:
    task = str(config.get("task", "reference_pca")).strip().lower()
    if task == "prediction_distribution":
        return run_prediction_distribution(config, run_dir)
    if task not in {"", "reference_pca"}:
        raise ValueError(f"Unsupported visualizers task: {task}")

    run_dir = Path(run_dir).resolve()
    inp = config.get("input", {}) or {}
    out_cfg = config.get("output", {}) or {}
    pca_cfg = out_cfg.get("pca", {}) or {}

    run_roots = _normalize_list(inp.get("run_roots", []), [])
    if not run_roots:
        raise ValueError("input.run_roots is required.")
    if len(run_roots) not in {1, 2}:
        raise ValueError("input.run_roots must contain 1 or 2 paths.")

    map_types = [v.lower() for v in _normalize_list(inp.get("map_types"), ["raw", "norm"])]
    groups = [v.lower() for v in _normalize_list(inp.get("groups"), ["noise", "fn", "non_fn"])]
    max_num = int(inp.get("max_num", 0))
    per_target = bool(pca_cfg.get("per_target", True))
    save_html = bool(pca_cfg.get("save_html", True))
    dimension = int(pca_cfg.get("dimension", 2))
    if not per_target:
        raise ValueError("output.pca.per_target must be true.")
    if not save_html:
        raise ValueError("output.pca.save_html must be true.")
    if dimension not in {2, 3}:
        raise ValueError("output.pca.dimension must be 2 or 3.")

    samples = collect_per_image_samples(
        run_roots=run_roots,
        map_types=map_types,
        groups=groups,
        repo_root=REPO_ROOT,
    )

    by_key: dict[tuple[str, str], list] = defaultdict(list)
    for s in samples:
        by_key[(s.target, s.map_type)].append(s)

    plots_root = run_dir / "plots" / f"pca_{dimension}d"
    results_root = run_dir / "results"
    plots_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "mode": "visualize",
        "input": {
            "run_roots": [str((REPO_ROOT / r).resolve()) if not Path(r).is_absolute() else str(Path(r).resolve()) for r in run_roots],
            "map_types": map_types,
            "groups": groups,
            "max_num": int(max_num),
        },
        "total_samples": int(len(samples)),
        "pca_dimension": int(dimension),
        "keys": {},
    }

    for key in sorted(by_key.keys()):
        target, map_type = key
        key_samples = _limit_samples_per_group(by_key[key], max_num=max_num)
        x, meta = build_key_matrix(key_samples)
        n_samples = int(x.shape[0])
        n_features = int(x.shape[1]) if x.ndim == 2 else 0
        reason = ""
        points = None

        if n_samples >= max(2, dimension) and n_features >= dimension:
            from sklearn.decomposition import PCA

            pca = PCA(n_components=dimension)
            points = pca.fit_transform(x)
            pc1 = points[:, 0]
            pc2 = points[:, 1]
            pc3 = points[:, 2] if dimension == 3 else np.full((n_samples,), np.nan, dtype=np.float32)
        else:
            reason = f"n_samples={n_samples}, n_features={n_features}"
            pc1 = np.full((n_samples,), np.nan, dtype=np.float32)
            pc2 = np.full((n_samples,), np.nan, dtype=np.float32)
            pc3 = np.full((n_samples,), np.nan, dtype=np.float32)

        rows = []
        for i, m in enumerate(meta):
            rows.append(
                {
                    "run_root": m["run_root"],
                    "group": m["group"],
                    "target": m["target"],
                    "map_type": m["map_type"],
                    "csv_path": m["csv_path"],
                    "vector_len": int(m["vector_len"]),
                    "pc1": float(pc1[i]) if i < len(pc1) and np.isfinite(pc1[i]) else np.nan,
                    "pc2": float(pc2[i]) if i < len(pc2) and np.isfinite(pc2[i]) else np.nan,
                    "pc3": float(pc3[i]) if i < len(pc3) and np.isfinite(pc3[i]) else np.nan,
                }
            )
        points_csv = results_root / f"pca_points_{target}_{map_type}.csv"
        pd.DataFrame(rows).to_csv(points_csv, index=False)

        plot_path = plots_root / target / f"{map_type}.html"
        save_pca_html(
            plot_path,
            points=points,
            groups=[m["group"] for m in meta],
            title=f"PCA {dimension}D - {target} ({map_type})",
            reason=reason,
            dimension=dimension,
        )

        group_counts = {}
        for g in [m["group"] for m in meta]:
            group_counts[g] = int(group_counts.get(g, 0) + 1)
        vec_lens = np.asarray([int(m["vector_len"]) for m in meta], dtype=np.int64) if meta else np.asarray([], dtype=np.int64)
        summary["keys"][f"{target}/{map_type}"] = {
            "num_samples": n_samples,
            "num_features": n_features,
            "group_counts": group_counts,
            "vector_len_min": int(vec_lens.min()) if vec_lens.size else 0,
            "vector_len_max": int(vec_lens.max()) if vec_lens.size else 0,
            "vector_len_mean": float(vec_lens.mean()) if vec_lens.size else 0.0,
            "points_csv": str(points_csv),
            "plot_html": str(plot_path),
            "pca_applied": bool(points is not None),
            "reason": reason,
        }

    if not by_key:
        no_data = run_dir / "results" / "summary_no_data.json"
        with open(no_data, "w", encoding="utf-8") as f:
            json.dump({"message": "No per-image CSV files were found."}, f, ensure_ascii=False, indent=2)

    return summary


def run_prediction_distribution(config: dict, run_dir: Path) -> dict:
    run_dir = Path(run_dir).resolve()
    inp = config.get("input", {}) or {}
    plots_cfg = config.get("plots", {}) or {}
    enabled = plots_cfg.get("enabled", None)
    max_rows = int(inp.get("max_rows", 0) or 0)

    df, source_summaries = collect_prediction_dump_data(
        csv_paths=_normalize_list(inp.get("csv_paths", []), []),
        run_roots=_normalize_list(inp.get("run_roots", []), []),
        repo_root=REPO_ROOT,
        max_rows=max_rows,
    )

    results_root = run_dir / "results"
    plots_root = run_dir / "plots" / "prediction_distribution"
    results_root.mkdir(parents=True, exist_ok=True)
    plots_root.mkdir(parents=True, exist_ok=True)

    merged_csv = results_root / "prediction_dump_merged.csv"
    df.to_csv(merged_csv, index=False)

    numeric_cols = [
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
        "obj",
        "cls_conf",
        "score",
        "obj_null_abs_diff",
        "cls_null_abs_diff",
        "score_null_abs_diff",
        "cls_entropy_norm",
        "cls_uniform_kl",
        "max_iou",
        "tp",
    ]
    summary_rows = []
    for col in numeric_cols:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if vals.empty:
            continue
        summary_rows.append(
            {
                "column": col,
                "count": int(vals.shape[0]),
                "min": float(vals.min()),
                "max": float(vals.max()),
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=0)),
                "p05": float(vals.quantile(0.05)),
                "p50": float(vals.quantile(0.50)),
                "p95": float(vals.quantile(0.95)),
            }
        )
    summary_stats_csv = results_root / "summary_stats.csv"
    pd.DataFrame(summary_rows).to_csv(summary_stats_csv, index=False)

    per_image_counts = (
        df.groupby(["source_csv", "image_id"], dropna=False).size().reset_index(name="num_predictions")
        if not df.empty and {"source_csv", "image_id"}.issubset(df.columns)
        else pd.DataFrame(columns=["source_csv", "image_id", "num_predictions"])
    )
    extra_count_rows = []
    for source in source_summaries:
        total_images = int(source.get("total_images", 0) or 0)
        if total_images <= 0:
            continue
        source_csv = source.get("csv_path", "")
        observed = int((per_image_counts["source_csv"] == source_csv).sum()) if not per_image_counts.empty else 0
        for i in range(max(0, total_images - observed)):
            extra_count_rows.append(
                {
                    "source_csv": source_csv,
                    "image_id": f"__no_prediction_{i}",
                    "num_predictions": 0,
                }
            )
    if extra_count_rows:
        per_image_counts = pd.concat([per_image_counts, pd.DataFrame(extra_count_rows)], ignore_index=True)
    per_image_counts_csv = results_root / "per_image_prediction_counts.csv"
    per_image_counts.to_csv(per_image_counts_csv, index=False)

    plot_outputs = save_prediction_distribution_plots(
        plots_root,
        df=df,
        enabled=enabled,
        image_counts=per_image_counts,
    )

    summary = {
        "mode": "visualize",
        "task": "prediction_distribution",
        "num_rows": int(df.shape[0]),
        "num_sources": int(len(source_summaries)),
        "sources": source_summaries,
        "merged_csv": str(merged_csv),
        "summary_stats_csv": str(summary_stats_csv),
        "per_image_counts_csv": str(per_image_counts_csv),
        "plots": plot_outputs,
    }
    return summary
