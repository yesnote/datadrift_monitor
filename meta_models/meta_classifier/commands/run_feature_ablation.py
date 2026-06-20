from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from losses.loss import evaluate_classifier
from models.meta_classifier import build_estimator
from meta_models.common import (
    FeatureSpec,
    build_feature_matrix,
    load_training_dataframe,
    sanitize_feature_matrix,
)


def _strip_feature_prefix(name: str) -> str:
    text = str(name)
    return text.split("__", 1)[1] if "__" in text else text


def _build_feature_groups(feature_columns: list[str]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    prob_cols = []
    for col in feature_columns:
        name = _strip_feature_prefix(col)
        if name.startswith("prob_") and name[5:].isdigit():
            prob_cols.append(col)
        else:
            groups.append({"feature": name, "feature_columns": [col]})

    if prob_cols:
        prob_cols = sorted(prob_cols, key=lambda c: int(_strip_feature_prefix(c).split("_", 1)[1]))
        insert_idx = 0
        for idx, group in enumerate(groups):
            if group["feature"] == "prob_sum":
                insert_idx = idx + 1
                break
        groups.insert(
            insert_idx,
            {
                "feature": "class_probability_vector",
                "feature_columns": prob_cols,
            },
        )
    return groups


def _evaluate_feature_set(
    df: pd.DataFrame,
    label_col: str,
    feature_columns: list[str],
    model_cfg: dict[str, Any],
    exp_cfg: dict[str, Any],
) -> dict[str, float]:
    spec = FeatureSpec(grad_columns=feature_columns, dim_by_column={c: 1 for c in feature_columns})
    x = build_feature_matrix(df, spec)
    x, _nonfinite_stats = sanitize_feature_matrix(x)
    y = df[label_col].astype(int).to_numpy()

    model_name = str(model_cfg.get("type", "gb_classifier"))
    device = str(model_cfg.get("device", "cpu"))
    random_seed = int(model_cfg.get("random_seed", 42))
    process = str(exp_cfg.get("process", "repeat")).strip().lower()

    aurocs: list[float] = []
    aps: list[float] = []

    if process == "kfold":
        kfold_cfg = exp_cfg.get("kfold", {})
        num_fold = int(kfold_cfg.get("num_fold", 5))
        splitter = StratifiedKFold(n_splits=num_fold, shuffle=True, random_state=random_seed)
        iterator = splitter.split(x, y)
    elif process == "repeat":
        repeat_cfg = exp_cfg.get("repeat", {})
        split = float(repeat_cfg.get("split", 0.3))
        repeats = int(repeat_cfg.get("repeats", 5))

        def _repeat_iter():
            for i in range(repeats):
                yield train_test_split(
                    np.arange(len(y)),
                    test_size=split,
                    random_state=random_seed + i,
                    stratify=y,
                    shuffle=True,
                )

        iterator = _repeat_iter()
    else:
        raise ValueError("experiment.process must be 'repeat' or 'kfold'.")

    for train_idx, test_idx in iterator:
        estimator = build_estimator(model_name, device=device, random_seed=random_seed)
        estimator.fit(x[train_idx], y[train_idx])
        y_pred = estimator.predict_proba(x[test_idx])[:, 1]
        auroc, ap = evaluate_classifier(y[test_idx], y_pred)
        aurocs.append(float(auroc))
        aps.append(float(ap))

    return {
        "auroc_mean": float(np.mean(aurocs)),
        "auroc_std": float(np.std(aurocs, ddof=1)) if len(aurocs) > 1 else 0.0,
        "ap_mean": float(np.mean(aps)),
        "ap_std": float(np.std(aps, ddof=1)) if len(aps) > 1 else 0.0,
    }


def _plot_single_feature(results_df: pd.DataFrame, out_path: Path) -> None:
    df = results_df.sort_values("auroc_mean", ascending=False).reset_index(drop=True)
    x = np.arange(len(df))
    fig_width = max(12.0, 0.18 * len(df) + 4.0)
    fig, ax = plt.subplots(figsize=(fig_width, 6.0))
    ax.plot(x, df["auroc_mean"], color="#4C78A8", linewidth=1.8, marker="o", markersize=2.5, label="AUROC")
    ax.plot(x, df["ap_mean"], color="#E45756", linewidth=1.8, marker="o", markersize=2.5, label="AP")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Classification performance")
    ax.set_xlabel("Feature")
    ax.set_title("Single MetaDetect Feature Performance")
    ax.set_xticks(x)
    ax.set_xticklabels(df["feature"], rotation=90, fontsize=7)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_cumulative(results_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    ax.plot(results_df["num_features"], results_df["auroc_mean"], label="AUROC", linewidth=2.0)
    ax.plot(results_df["num_features"], results_df["ap_mean"], label="AP", linewidth=2.0)
    y_values = np.concatenate(
        [
            results_df["auroc_mean"].to_numpy(dtype=np.float64),
            results_df["ap_mean"].to_numpy(dtype=np.float64),
        ]
    )
    y_values = y_values[np.isfinite(y_values)]
    if y_values.size:
        y_min = float(np.min(y_values))
        y_max = float(np.max(y_values))
        y_span = max(y_max - y_min, 0.02)
        pad = max(0.01, y_span * 0.15)
        ax.set_ylim(max(0.0, y_min - pad), min(1.0, y_max + pad))
    else:
        ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Number of features")
    ax.set_ylabel("Classification performance")
    ax.set_title("Cumulative MetaDetect Feature Performance")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_cumulative_delta(results_df: pd.DataFrame, out_path: Path) -> None:
    if len(results_df) <= 1:
        fig, ax = plt.subplots(figsize=(8.0, 5.5))
        ax.text(0.5, 0.5, "Need at least two features for delta plot", ha="center", va="center")
        ax.set_axis_off()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        return

    df = results_df.copy().reset_index(drop=True)
    df["delta_auroc"] = df["auroc_mean"].diff()
    df["delta_ap"] = df["ap_mean"].diff()
    df = df.iloc[1:].reset_index(drop=True)

    x = np.arange(len(df))
    fig_width = max(10.0, 0.22 * len(df) + 4.0)
    fig, ax = plt.subplots(figsize=(fig_width, 5.5))
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.2)
    ax.plot(x, df["delta_auroc"], label="Delta AUROC", color="#4C78A8", linewidth=1.8, marker="o", markersize=3)
    ax.plot(x, df["delta_ap"], label="Delta AP", color="#E45756", linewidth=1.8, marker="o", markersize=3)

    y_values = np.concatenate(
        [
            df["delta_auroc"].to_numpy(dtype=np.float64),
            df["delta_ap"].to_numpy(dtype=np.float64),
        ]
    )
    y_values = y_values[np.isfinite(y_values)]
    if y_values.size:
        y_abs = max(float(np.max(np.abs(y_values))), 0.01)
        ax.set_ylim(-y_abs * 1.2, y_abs * 1.2)

    ax.set_xlabel("Added feature")
    ax.set_ylabel("Performance delta")
    ax.set_title("Cumulative MetaDetect Feature Marginal Delta")
    ax.set_xticks(x)
    ax.set_xticklabels(df["added_feature"], rotation=90, fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_feature_ablation(config: dict[str, Any], run_dir: Path) -> Path:
    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    exp_cfg = config.get("experiment", {})

    out_dir = Path(run_dir).resolve()
    results_dir = out_dir / "results"
    plots_dir = out_dir / "plots"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    df, label_col, feature_columns, root_info = load_training_dataframe(dataset_cfg, task="classifier")
    if root_info.get("input_uncertainty") != ["meta_detect"]:
        raise ValueError("feature_ablation currently expects exactly one meta_detect input_root.")

    feature_groups = _build_feature_groups(feature_columns)

    single_rows = []
    for group in tqdm(feature_groups, desc="Single feature ablation", unit="feature"):
        group_columns = list(group["feature_columns"])
        metrics = _evaluate_feature_set(df, label_col, group_columns, model_cfg, exp_cfg)
        single_rows.append(
            {
                "feature": group["feature"],
                "num_columns": len(group_columns),
                "feature_columns": json.dumps(group_columns, ensure_ascii=False),
                **metrics,
            }
        )
    single_df = pd.DataFrame(single_rows).sort_values("auroc_mean", ascending=False).reset_index(drop=True)
    single_df.to_csv(results_dir / "single_feature_results.csv", index=False)

    group_by_name = {group["feature"]: list(group["feature_columns"]) for group in feature_groups}
    ordered_feature_names = single_df["feature"].tolist()
    cumulative_rows = []
    selected: list[str] = []
    for feature_name in tqdm(ordered_feature_names, desc="Cumulative feature ablation", unit="feature"):
        selected.extend(group_by_name[feature_name])
        metrics = _evaluate_feature_set(df, label_col, selected, model_cfg, exp_cfg)
        cumulative_rows.append(
            {
                "num_features": len(cumulative_rows) + 1,
                "num_columns": len(selected),
                "added_feature": feature_name,
                "feature_columns": json.dumps(selected, ensure_ascii=False),
                **metrics,
            }
        )
    cumulative_df = pd.DataFrame(cumulative_rows)
    cumulative_df.to_csv(results_dir / "cumulative_feature_results.csv", index=False)

    _plot_single_feature(single_df, plots_dir / "single_feature_performance.png")
    _plot_cumulative(cumulative_df, plots_dir / "cumulative_feature_performance.png")
    _plot_cumulative_delta(cumulative_df, plots_dir / "cumulative_feature_delta_performance.png")

    metadata = {
        "input_root": root_info["input_root"],
        "gt_root": root_info["gt_root"],
        "label_column": label_col,
        "model": str(model_cfg.get("type", "gb_classifier")),
        "feature_dimension": len(feature_columns),
        "feature_group_count": len(feature_groups),
        "num_rows": int(len(df)),
        "num_positive_tp": int(df[label_col].astype(int).sum()),
        "process": str(exp_cfg.get("process", "repeat")),
        "input_features": feature_columns,
        "feature_groups": feature_groups,
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Saved feature ablation outputs to: {out_dir}")
    return out_dir
