from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


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


def save_eval_plots(y_true: np.ndarray, y_score: np.ndarray, out_dir: Path, model_name: str) -> None:
    plot_dir = out_dir / "plots" / model_name
    plot_dir.mkdir(parents=True, exist_ok=True)
    _save_curve_roc(y_true, y_score, plot_dir / "roc_curve.png")
    _save_curve_pr(y_true, y_score, plot_dir / "pr_curve.png")
    _save_score_distribution(y_true, y_score, plot_dir / "score_distribution_fn_vs_nonfn.png")
    _save_precision_at_k(y_true, y_score, plot_dir / "precision_at_k.png")

