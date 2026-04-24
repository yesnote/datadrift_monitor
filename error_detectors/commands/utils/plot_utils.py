from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
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


def _save_score_distribution(
    y_true: np.ndarray,
    y_score: np.ndarray,
    out_path: Path,
    pos_name: str,
    neg_name: str,
) -> None:
    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots(figsize=(6, 5))
        pos_scores = y_score[y_true == 1]
        neg_scores = y_score[y_true == 0]

        bins = 40
        color_neg = "#4C78A8"
        color_pos = "#E45756"
        if neg_scores.size > 0:
            sns.histplot(
                neg_scores,
                bins=bins,
                stat="density",
                alpha=0.45,
                element="step",
                fill=True,
                color=color_neg,
                ax=ax,
                label=f"{neg_name} (n={neg_scores.size})",
            )
        if pos_scores.size > 0:
            sns.histplot(
                pos_scores,
                bins=bins,
                stat="density",
                alpha=0.45,
                element="step",
                fill=True,
                color=color_pos,
                ax=ax,
                label=f"{pos_name} (n={pos_scores.size})",
            )
        if pos_scores.size == 0 and neg_scores.size == 0:
            ax.text(0.5, 0.5, "No scores to plot", ha="center", va="center")

        ax.set_facecolor("#f7f7f7")
        ax.grid(True, which="major", axis="both", color="#d9d9d9", linewidth=0.8, alpha=0.8)
        ax.set_title(f"{pos_name} vs {neg_name} Score Distribution")
        ax.set_xlabel(f"Predicted {pos_name} score")
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


def _save_confidence_vs_conditional_precision(y_true: np.ndarray, y_score: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    scores = np.clip(np.asarray(y_score, dtype=np.float64).reshape(-1), 0.0, 1.0)
    labels = np.asarray(y_true, dtype=np.float64).reshape(-1)
    bins = np.linspace(0.0, 1.0, 21, dtype=np.float64)
    centers: list[float] = []
    precision_vals: list[float] = []
    for i in range(len(bins) - 1):
        lo = bins[i]
        hi = bins[i + 1]
        if i == len(bins) - 2:
            mask = (scores >= lo) & (scores <= hi)
        else:
            mask = (scores >= lo) & (scores < hi)
        if not np.any(mask):
            continue
        centers.append(float((lo + hi) * 0.5))
        precision_vals.append(float(labels[mask].mean()))

    ax.plot([0.0, 1.0], [0.0, 1.0], color="gray", linestyle="--", linewidth=1.5, label="Oracle")
    if centers:
        ax.plot(centers, precision_vals, color="#1f77b4", linewidth=2.0, marker="o", markersize=3.5, label="Observed")
    else:
        ax.text(0.5, 0.5, "No samples", ha="center", va="center")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Confidence vs Conditional Precision")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Conditional Precision")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_pca_2d(
    x_features: np.ndarray,
    y_true: np.ndarray,
    estimator,
    out_path: Path,
) -> None:
    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots(figsize=(8, 6))
        try:
            x_arr = np.asarray(x_features, dtype=np.float64)
            y_arr = np.asarray(y_true, dtype=np.int64).reshape(-1)
            if x_arr.ndim != 2 or x_arr.shape[0] < 2 or x_arr.shape[1] < 2:
                raise ValueError("PCA 2D requires at least 2 samples and 2 features.")

            pca = PCA(n_components=2)
            x_2d = pca.fit_transform(x_arr)

            x_min = float(x_2d[:, 0].min() - 1.0)
            x_max = float(x_2d[:, 0].max() + 1.0)
            y_min = float(x_2d[:, 1].min() - 1.0)
            y_max = float(x_2d[:, 1].max() + 1.0)
            xx, yy = np.meshgrid(
                np.linspace(x_min, x_max, 300, dtype=np.float64),
                np.linspace(y_min, y_max, 300, dtype=np.float64),
            )
            grid_2d = np.c_[xx.ravel(), yy.ravel()]
            grid_original = pca.inverse_transform(grid_2d)
            z = estimator.predict_proba(grid_original)[:, 1].reshape(xx.shape)

            contour = ax.contourf(xx, yy, z, levels=50, cmap="coolwarm", alpha=0.35)
            fig.colorbar(contour, ax=ax, label="Predicted Positive Probability")
            ax.contour(xx, yy, z, levels=[0.5], colors="black", linewidths=1.8)

            labels = np.where(y_arr == 1, "positive", "negative")
            sns.scatterplot(
                x=x_2d[:, 0],
                y=x_2d[:, 1],
                hue=labels,
                hue_order=["negative", "positive"],
                palette={"negative": "#4C78A8", "positive": "#E45756"},
                edgecolor="black",
                s=45,
                alpha=0.85,
                ax=ax,
            )
            ax.set_title("PCA 2D Decision Boundary")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.legend(title="Class", loc="best")
        except Exception as e:  # pragma: no cover - depends on data shape and estimator
            ax.text(0.5, 0.5, f"PCA 2D unavailable\n{e}", ha="center", va="center")
            ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


def save_eval_plots(
    y_true: np.ndarray,
    y_score: np.ndarray,
    out_dir: Path,
    model_name: str,
    label_col: str = "fn",
    x_features: np.ndarray | None = None,
    estimator=None,
) -> None:
    plot_dir = out_dir / "plots" / model_name
    plot_dir.mkdir(parents=True, exist_ok=True)
    label_key = str(label_col).strip().lower()
    if label_key == "tp":
        pos_name = "FP"
        neg_name = "non-FP"
        dist_name = "score_distribution_fp_vs_nonfp.png"
    else:
        pos_name = "FN"
        neg_name = "non-FN"
        dist_name = "score_distribution_fn_vs_nonfn.png"
    _save_curve_roc(y_true, y_score, plot_dir / "roc_curve.png")
    _save_curve_pr(y_true, y_score, plot_dir / "pr_curve.png")
    _save_score_distribution(y_true, y_score, plot_dir / dist_name, pos_name=pos_name, neg_name=neg_name)
    _save_precision_at_k(y_true, y_score, plot_dir / "precision_at_k.png")
    if label_key == "tp":
        _save_confidence_vs_conditional_precision(
            y_true,
            y_score,
            plot_dir / "confidence_vs_conditional_precision.png",
        )
    if x_features is not None and estimator is not None:
        _save_pca_2d(
            x_features=np.asarray(x_features),
            y_true=np.asarray(y_true),
            estimator=estimator,
            out_path=plot_dir / "pca_2d.png",
        )
