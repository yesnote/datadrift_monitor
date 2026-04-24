from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from utils.loader import build_key_matrix, collect_per_image_samples
from utils.plot import save_pca_plot

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
    save_png = bool(pca_cfg.get("save_png", True))
    if not per_target:
        raise ValueError("output.pca.per_target must be true.")
    if not save_png:
        raise ValueError("output.pca.save_png must be true.")

    samples = collect_per_image_samples(
        run_roots=run_roots,
        map_types=map_types,
        groups=groups,
        repo_root=REPO_ROOT,
    )

    by_key: dict[tuple[str, str], list] = defaultdict(list)
    for s in samples:
        by_key[(s.target, s.map_type)].append(s)

    plots_root = run_dir / "plots" / "pca_2d"
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

        if n_samples >= 2 and n_features >= 2:
            pca = PCA(n_components=2)
            points = pca.fit_transform(x)
            pc1 = points[:, 0]
            pc2 = points[:, 1]
        else:
            reason = f"n_samples={n_samples}, n_features={n_features}"
            pc1 = np.full((n_samples,), np.nan, dtype=np.float32)
            pc2 = np.full((n_samples,), np.nan, dtype=np.float32)

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
                }
            )
        points_csv = results_root / f"pca_points_{target}_{map_type}.csv"
        pd.DataFrame(rows).to_csv(points_csv, index=False)

        plot_path = plots_root / target / f"{map_type}.png"
        save_pca_plot(
            plot_path,
            points_2d=points,
            groups=[m["group"] for m in meta],
            title=f"PCA 2D - {target} ({map_type})",
            reason=reason,
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
            "plot_png": str(plot_path),
            "pca_applied": bool(points is not None),
            "reason": reason,
        }

    if not by_key:
        no_data = run_dir / "results" / "summary_no_data.json"
        with open(no_data, "w", encoding="utf-8") as f:
            json.dump({"message": "No per-image CSV files were found."}, f, ensure_ascii=False, indent=2)

    return summary
