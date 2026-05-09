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
        "anchor_w",
        "anchor_h",
        "anchor_area",
        "bbox_anchor_center_l2",
        "bbox_anchor_log_w_ratio",
        "bbox_anchor_log_h_ratio",
        "bbox_anchor_log_area_ratio",
        "bbox_anchor_aspect_ratio_diff",
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
    analysis_features = [
        "score",
        "obj",
        "cls_conf",
        "obj_null_abs_diff",
        "cls_null_abs_diff",
        "score_null_abs_diff",
        "cls_entropy_norm",
        "cls_uniform_kl",
        "bbox_anchor_center_l2",
        "bbox_anchor_log_w_ratio",
        "bbox_anchor_log_h_ratio",
        "bbox_anchor_log_area_ratio",
        "bbox_anchor_aspect_ratio_diff",
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

    tp_fp_csv = results_root / "tp_fp_feature_comparison.csv"
    detection_metrics_csv = results_root / "fp_detection_metrics.csv"
    class_metrics_csv = results_root / "classwise_fp_detection_metrics.csv"
    high_score_fp_csv = results_root / "high_score_fp_analysis.csv"
    _save_uncertainty_analysis_tables(
        df=df,
        features=analysis_features,
        tp_fp_csv=tp_fp_csv,
        detection_metrics_csv=detection_metrics_csv,
        class_metrics_csv=class_metrics_csv,
        high_score_fp_csv=high_score_fp_csv,
    )

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
        "tp_fp_feature_comparison_csv": str(tp_fp_csv),
        "fp_detection_metrics_csv": str(detection_metrics_csv),
        "classwise_fp_detection_metrics_csv": str(class_metrics_csv),
        "high_score_fp_analysis_csv": str(high_score_fp_csv),
        "per_image_counts_csv": str(per_image_counts_csv),
        "plots": plot_outputs,
    }
    return summary


def _clean_xy(feature_values, labels):
    x = pd.to_numeric(feature_values, errors="coerce").to_numpy(dtype=np.float64, copy=False)
    y = pd.to_numeric(labels, errors="coerce").to_numpy(dtype=np.float64, copy=False)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask].astype(np.int64)


def _rankdata_average(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.shape[0], dtype=np.float64)
    i = 0
    while i < x.shape[0]:
        j = i + 1
        while j < x.shape[0] and x[order[j]] == x[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata_average(scores)
    rank_sum_pos = float(ranks[pos].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    pos_total = int((labels == 1).sum())
    if pos_total == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    tp_cum = np.cumsum(y == 1)
    fp_cum = np.cumsum(y == 0)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)
    recall = tp_cum / float(pos_total)
    prev_recall = np.concatenate(([0.0], recall[:-1]))
    return float(np.sum((recall - prev_recall) * precision))


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.shape[0] < 2:
        return float("nan")
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std <= 0 or y_std <= 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _effect_size(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    var_a = float(np.var(a))
    var_b = float(np.var(b))
    pooled = ((var_a + var_b) / 2.0) ** 0.5
    if pooled <= 0:
        return float("nan")
    return (float(np.mean(a)) - float(np.mean(b))) / pooled


def _operating_points(scores: np.ndarray, labels: np.ndarray) -> dict:
    pos_total = int((labels == 1).sum())
    neg_total = int((labels == 0).sum())
    out = {
        "threshold_at_tpr95": np.nan,
        "fpr_at_tpr95": np.nan,
        "threshold_at_fpr05": np.nan,
        "tpr_at_fpr05": np.nan,
    }
    if pos_total == 0 or neg_total == 0:
        return out
    order = np.argsort(-scores, kind="mergesort")
    s = scores[order]
    y = labels[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    tpr = tp / float(pos_total)
    fpr = fp / float(neg_total)
    idx = np.where(tpr >= 0.95)[0]
    if idx.size:
        i = int(idx[0])
        out["threshold_at_tpr95"] = float(s[i])
        out["fpr_at_tpr95"] = float(fpr[i])
    idx = np.where(fpr <= 0.05)[0]
    if idx.size:
        i = int(idx[-1])
        out["threshold_at_fpr05"] = float(s[i])
        out["tpr_at_fpr05"] = float(tpr[i])
    return out


def _feature_metric_rows(df: pd.DataFrame, features: list[str]) -> tuple[list[dict], list[dict]]:
    if df.empty or "tp" not in df.columns:
        return [], []
    fp_label = 1 - pd.to_numeric(df["tp"], errors="coerce")
    tp_fp_rows = []
    metric_rows = []
    max_iou = pd.to_numeric(df["max_iou"], errors="coerce") if "max_iou" in df.columns else None

    for feature in features:
        if feature not in df.columns:
            continue
        values = pd.to_numeric(df[feature], errors="coerce")
        x, y_fp = _clean_xy(values, fp_label)
        if x.size == 0:
            continue
        tp_vals = x[y_fp == 0]
        fp_vals = x[y_fp == 1]
        tp_fp_rows.append(
            {
                "feature": feature,
                "n": int(x.size),
                "tp_n": int(tp_vals.size),
                "fp_n": int(fp_vals.size),
                "tp_mean": float(np.mean(tp_vals)) if tp_vals.size else np.nan,
                "fp_mean": float(np.mean(fp_vals)) if fp_vals.size else np.nan,
                "fp_minus_tp_mean": (float(np.mean(fp_vals)) - float(np.mean(tp_vals))) if fp_vals.size and tp_vals.size else np.nan,
                "tp_median": float(np.median(tp_vals)) if tp_vals.size else np.nan,
                "fp_median": float(np.median(fp_vals)) if fp_vals.size else np.nan,
                "cohen_d_fp_minus_tp": _effect_size(fp_vals, tp_vals),
            }
        )
        corr_iou = np.nan
        if max_iou is not None:
            xi, iou = _clean_xy(values, max_iou)
            corr_iou = _pearson(xi, iou.astype(np.float64))
        ops = _operating_points(x, y_fp)
        metric_rows.append(
            {
                "feature": feature,
                "n": int(x.size),
                "fp_positive_rate": float(np.mean(y_fp)) if y_fp.size else np.nan,
                "auroc_fp_high": _auroc(x, y_fp),
                "auprc_fp_high": _average_precision(x, y_fp),
                "auroc_fp_low": _auroc(-x, y_fp),
                "auprc_fp_low": _average_precision(-x, y_fp),
                "corr_with_max_iou": corr_iou,
                **ops,
            }
        )
    return tp_fp_rows, metric_rows


def _save_uncertainty_analysis_tables(
    *,
    df: pd.DataFrame,
    features: list[str],
    tp_fp_csv: Path,
    detection_metrics_csv: Path,
    class_metrics_csv: Path,
    high_score_fp_csv: Path,
) -> None:
    tp_fp_rows, metric_rows = _feature_metric_rows(df, features)
    pd.DataFrame(tp_fp_rows).to_csv(tp_fp_csv, index=False)
    pd.DataFrame(metric_rows).to_csv(detection_metrics_csv, index=False)

    class_rows = []
    if not df.empty and "pred_class" in df.columns and "tp" in df.columns:
        for cls_name, group in df.groupby(df["pred_class"].astype(str), dropna=False):
            if int(group.shape[0]) < 20:
                continue
            _tp_fp, metrics = _feature_metric_rows(group, features)
            for row in metrics:
                row = dict(row)
                row["pred_class"] = cls_name
                row["class_n"] = int(group.shape[0])
                class_rows.append(row)
    pd.DataFrame(class_rows).to_csv(class_metrics_csv, index=False)

    high_score_rows = []
    if not df.empty and {"score", "tp"}.issubset(df.columns):
        score = pd.to_numeric(df["score"], errors="coerce")
        tp = pd.to_numeric(df["tp"], errors="coerce")
        for threshold in [0.25, 0.5, 0.75, 0.9]:
            subset = df[(score >= threshold) & np.isfinite(score) & np.isfinite(tp)]
            if subset.empty:
                continue
            fp_rate = float((pd.to_numeric(subset["tp"], errors="coerce") == 0).mean())
            row = {
                "score_threshold": threshold,
                "n": int(subset.shape[0]),
                "fp_rate": fp_rate,
            }
            for feature in features:
                if feature not in subset.columns:
                    continue
                vals = pd.to_numeric(subset[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                if vals.empty:
                    continue
                row[f"{feature}_mean"] = float(vals.mean())
                row[f"{feature}_median"] = float(vals.median())
            high_score_rows.append(row)
    pd.DataFrame(high_score_rows).to_csv(high_score_fp_csv, index=False)
