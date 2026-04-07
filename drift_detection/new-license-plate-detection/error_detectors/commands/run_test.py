from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve

from error_detectors.losses.loss import evaluate_classifier
from error_detectors.commands.run_train import (
    FeatureSpec,
    build_feature_matrix,
    infer_feature_spec,
    load_object,
    load_training_dataframe,
    sanitize_feature_matrix,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _parse_model_index(path: Path) -> int:
    match = re.match(r"^model_(\d+)\.joblib$", path.name)
    if not match:
        return 10**9
    return int(match.group(1))


def _resolve_model_paths(model_cfg: dict[str, Any]) -> list[Path]:
    model_root_raw = str(model_cfg.get("model_root", "")).strip()
    if not model_root_raw:
        raise ValueError("For mode='test', set model.model_root in config.")
    model_root = Path(model_root_raw).resolve()
    if not model_root.is_dir():
        raise FileNotFoundError(f"model.model_root not found: {model_root}")

    paths = [*model_root.glob("model_*.joblib")]
    paths = sorted({p.resolve() for p in paths}, key=_parse_model_index)
    if not paths:
        raise FileNotFoundError(f"No model_*.joblib files found in {model_root}")
    return paths


def _load_feature_spec_for_test(model_root: Path, df: pd.DataFrame, current_features: list[str]) -> FeatureSpec:
    metadata_path = model_root / "metadata.json"
    if metadata_path.is_file():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        input_features = metadata.get("input_features")
        dim_by_feature = metadata.get("dim_by_feature")
        if isinstance(input_features, list) and isinstance(dim_by_feature, dict):
            missing = [c for c in input_features if c not in df.columns]
            if missing:
                raise ValueError(
                    "Missing required feature columns from training metadata: "
                    f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
                )
            dim_dict = {str(k): int(v) for k, v in dim_by_feature.items() if str(k) in input_features}
            return FeatureSpec(grad_columns=[str(c) for c in input_features], dim_by_column=dim_dict)
    return infer_feature_spec(df, current_features)


def _save_curve_roc(y_true: np.ndarray, y_score: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    try:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        ax.plot(fpr, tpr, lw=2, label="ROC")
        ax.plot([0, 1], [0, 1], linestyle="--", lw=1, color="gray", label="Random")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.legend(loc="lower right")
    except Exception as e:  # pragma: no cover - depends on label distribution
        ax.text(0.5, 0.5, f"ROC unavailable\n{e}", ha="center", va="center")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_curve_pr(y_true: np.ndarray, y_score: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        ax.plot(recall, precision, lw=2, label=f"PR (AP={ap:.4f})")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.legend(loc="lower left")
    except Exception as e:  # pragma: no cover - depends on label distribution
        ax.text(0.5, 0.5, f"PR unavailable\n{e}", ha="center", va="center")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_score_distribution(y_true: np.ndarray, y_score: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    fn_scores = y_score[y_true == 1]
    non_fn_scores = y_score[y_true == 0]

    bins = 40
    if non_fn_scores.size > 0:
        ax.hist(non_fn_scores, bins=bins, alpha=0.6, density=True, label=f"non-FN (n={non_fn_scores.size})")
    if fn_scores.size > 0:
        ax.hist(fn_scores, bins=bins, alpha=0.6, density=True, label=f"FN (n={fn_scores.size})")
    if fn_scores.size == 0 and non_fn_scores.size == 0:
        ax.text(0.5, 0.5, "No scores to plot", ha="center", va="center")

    ax.set_title("FN vs non-FN Score Distribution")
    ax.set_xlabel("Predicted FN score")
    ax.set_ylabel("Density")
    ax.legend(loc="upper center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_precision_at_k(y_true: np.ndarray, y_score: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    n = int(y_true.shape[0])
    if n == 0:
        ax.text(0.5, 0.5, "No samples", ha="center", va="center")
    else:
        order = np.argsort(-y_score)
        y_sorted = y_true[order].astype(np.float64)
        cum_tp = np.cumsum(y_sorted)
        k_all = np.arange(1, n + 1, dtype=np.int64)
        precision_all = cum_tp / k_all

        max_points = 1000
        if n > max_points:
            idx = np.unique(np.linspace(0, n - 1, num=max_points, dtype=np.int64))
            k_plot = k_all[idx]
            precision_plot = precision_all[idx]
        else:
            k_plot = k_all
            precision_plot = precision_all

        ax.plot(k_plot, precision_plot, lw=2)
        ax.set_xlim(1, n)
        ax.set_ylim(0.0, 1.0)

    ax.set_title("Precision@K")
    ax.set_xlabel("K (top-K by predicted FN score)")
    ax.set_ylabel("Precision")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_eval_plots(y_true: np.ndarray, y_score: np.ndarray, out_dir: Path, model_name: str) -> None:
    plot_dir = out_dir / "plots" / model_name
    plot_dir.mkdir(parents=True, exist_ok=True)
    _save_curve_roc(y_true, y_score, plot_dir / "roc_curve.png")
    _save_curve_pr(y_true, y_score, plot_dir / "pr_curve.png")
    _save_score_distribution(y_true, y_score, plot_dir / "score_distribution_fn_vs_nonfn.png")
    _save_precision_at_k(y_true, y_score, plot_dir / "precision_at_k.png")


def run_test(config: dict[str, Any], run_dir: Path) -> Path:
    dataset_cfg = config["dataset"]
    model_cfg = config["model"]

    out_dir = run_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model_paths = _resolve_model_paths(model_cfg)
    model_root = model_paths[0].parent

    df, label_col, input_features, root_info = load_training_dataframe(dataset_cfg)
    y = df[label_col].astype(int).to_numpy()
    spec = _load_feature_spec_for_test(model_root, df, input_features)
    x = build_feature_matrix(df, spec)
    x, nonfinite_stats = sanitize_feature_matrix(x)
    had_nonfinite = any(v > 0 for v in nonfinite_stats.values())

    eval_rows: list[dict[str, float | int | str]] = []
    for model_path in model_paths:
        estimator = load_object(model_path)
        y_pred = estimator.predict_proba(x)[:, 1]
        auroc, ap = evaluate_classifier(y, y_pred)
        model_name = model_path.stem
        eval_rows.append(
            {
                "model_file": model_path.name,
                "model_index": int(_parse_model_index(model_path)) if _parse_model_index(model_path) < 10**9 else -1,
                "auroc": float(auroc),
                "ap": float(ap),
            }
        )
        pd.DataFrame({"y_test": y, "y_pred": y_pred}).to_csv(
            out_dir / f"eval_data_test_{model_name}.csv",
            index=False,
        )
        _save_eval_plots(y, y_pred, out_dir=out_dir, model_name=model_name)

    eval_df = pd.DataFrame(eval_rows)
    summary = {
        "model_file": "mean",
        "model_index": -1,
        "auroc": float(eval_df["auroc"].mean()),
        "ap": float(eval_df["ap"].mean()),
    }
    summary_std = {
        "model_file": "std",
        "model_index": -1,
        "auroc": float(eval_df["auroc"].std(ddof=0)),
        "ap": float(eval_df["ap"].std(ddof=0)),
    }
    eval_df = pd.concat([eval_df, pd.DataFrame([summary, summary_std])], ignore_index=True)
    eval_df.to_csv(out_dir / "evaluation_results.csv", index=False)

    metadata_test = {
        "mode": "test",
        "loaded_models": [str(p) for p in model_paths],
        "source_model_root": str(model_root),
        "input_root": root_info["input_root"],
        "gt_root": root_info["gt_root"],
        "model_group": root_info["model_group"],
        "input_uncertainty": root_info["input_uncertainty"],
        "input_target": root_info.get("input_target", []),
        "label_column": label_col,
        "num_rows": int(len(df)),
        "num_positive": int(np.sum(y)),
        "feature_dimension": int(x.shape[1]),
        "input_features": spec.grad_columns,
        "dim_by_feature": spec.dim_by_column,
        "nonfinite_replaced": had_nonfinite,
        "nonfinite_stats": nonfinite_stats,
    }
    with open(out_dir / "metadata_test.json", "w", encoding="utf-8") as f:
        json.dump(metadata_test, f, ensure_ascii=False, indent=2)

    print(f"Saved outputs to: {out_dir}")
    print(eval_df)
    return out_dir
