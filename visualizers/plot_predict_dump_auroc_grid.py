from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

# Fill this with one or more object_detectors/runs/.../<time>_predict_dump paths.
PREDICT_DUMP_ROOTS = [
    r"object_detectors/runs/predict/coco/00-00-0000_00;00_predict_dump",
]

OUTPUT_ROOT = Path(__file__).resolve().parent / "runs"


def _resolve_dump_csv(root):
    root = Path(root)
    if root.is_file():
        return root
    return root / "predict_dump.csv"


def _load_predict_dump():
    frames = []
    for root in PREDICT_DUMP_ROOTS:
        csv_path = _resolve_dump_csv(root)
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing predict_dump.csv: {csv_path}")
        df = pd.read_csv(csv_path)
        df["source_run"] = str(Path(root))
        frames.append(df)
    if not frames:
        raise ValueError("PREDICT_DUMP_ROOTS is empty.")
    df = pd.concat(frames, ignore_index=True)
    if "tp" not in df.columns:
        raise ValueError("predict_dump.csv must contain a 'tp' column.")
    df["tp"] = pd.to_numeric(df["tp"], errors="coerce").fillna(0).astype(int)
    return df


def _numeric_score(df, column):
    if column not in df.columns:
        raise ValueError(f"Missing required column: {column}")
    return pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)


def _bbox_area_score(df):
    if "bbox_area" in df.columns:
        return _numeric_score(df, "bbox_area")
    for column in ("bbox_w", "bbox_h"):
        if column not in df.columns:
            raise ValueError(f"Missing required column: {column}")
    w = pd.to_numeric(df["bbox_w"], errors="coerce").to_numpy(dtype=float)
    h = pd.to_numeric(df["bbox_h"], errors="coerce").to_numpy(dtype=float)
    return np.maximum(w, 0.0) * np.maximum(h, 0.0)


def _oriented_roc(y, score):
    valid = np.isfinite(score)
    y_valid = y[valid]
    score_valid = score[valid]
    if len(y_valid) == 0 or len(np.unique(y_valid)) < 2:
        return None
    auc = roc_auc_score(y_valid, score_valid)
    direction = "TP"
    plot_score = score_valid
    if auc < 0.5:
        plot_score = -score_valid
        auc = roc_auc_score(y_valid, plot_score)
        direction = "FP"
    fpr, tpr, _ = roc_curve(y_valid, plot_score)
    return {
        "fpr": fpr,
        "tpr": tpr,
        "auc": float(auc),
        "direction": direction,
        "n": int(len(y_valid)),
    }


def _plot_grid(df, out_dir):
    y = df["tp"].to_numpy(dtype=int)
    panels = [
        ("Prediction", "score", "Score", lambda d: _numeric_score(d, "score")),
        (
            "Prediction",
            "objectness",
            "Objectness",
            lambda d: _numeric_score(d, "objectness"),
        ),
        (
            "Prediction",
            "class_probability",
            "Class probability",
            lambda d: _numeric_score(d, "class_probability"),
        ),
        ("Prediction", "bbox_area", "BBox area", _bbox_area_score),
        (
            "Cand target loss",
            "score_cand_diff",
            "Score diff",
            lambda d: _numeric_score(d, "score_cand_diff"),
        ),
        (
            "Cand target loss",
            "obj_cand_bce_loss",
            "Objectness BCE",
            lambda d: _numeric_score(d, "obj_cand_bce_loss"),
        ),
        (
            "Cand target loss",
            "cls_cand_onehot_bce_loss",
            "Class one-hot BCE",
            lambda d: _numeric_score(d, "cls_cand_onehot_bce_loss"),
        ),
        (
            "Cand target loss",
            "bbox_cand_ciou_loss",
            "BBox CIoU loss",
            lambda d: _numeric_score(d, "bbox_cand_ciou_loss"),
        ),
        (
            "Null target loss",
            "score_null_diff",
            "Score diff",
            lambda d: _numeric_score(d, "score_null_diff"),
        ),
        (
            "Null target loss",
            "obj_null_bce_loss",
            "Objectness BCE",
            lambda d: _numeric_score(d, "obj_null_bce_loss"),
        ),
        (
            "Null target loss",
            "cls_uniform_kl",
            "Class KL to uniform",
            lambda d: _numeric_score(d, "cls_uniform_kl"),
        ),
        (
            "Null target loss",
            "bbox_null_ciou_loss",
            "BBox CIoU loss",
            lambda d: _numeric_score(d, "bbox_null_ciou_loss"),
        ),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(17, 11), sharex=True, sharey=True)
    rows = []
    for ax, (row_name, feature, title, score_fn) in zip(axes.ravel(), panels):
        score = score_fn(df)
        result = _oriented_roc(y, score)
        ax.plot([0, 1], [0, 1], color="0.75", linewidth=1, linestyle="--")
        if result is None:
            ax.text(
                0.5,
                0.5,
                "not enough data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            auc = np.nan
            direction = ""
            n = int(np.isfinite(score).sum())
        else:
            ax.plot(result["fpr"], result["tpr"], color="#2563eb", linewidth=2.0)
            auc = result["auc"]
            direction = result["direction"]
            n = result["n"]
            ax.text(
                0.05,
                0.08,
                f"AUROC {auc:.3f}\nhigh={direction}",
                transform=ax.transAxes,
                fontsize=10,
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": "0.85",
                    "alpha": 0.9,
                },
            )
        ax.set_title(title, fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, color="0.9", linewidth=0.8)
        rows.append(
            {
                "group": row_name,
                "feature": feature,
                "auroc": auc,
                "high_value_indicates": direction,
                "n": n,
            }
        )

    for row_idx, row_name in enumerate(
        ["Prediction", "Cand target loss", "Null target loss"]
    ):
        axes[row_idx, 0].set_ylabel(f"{row_name}\nTPR")
    for ax in axes[-1, :]:
        ax.set_xlabel("FPR")
    fig.suptitle("TP/FP AUROC from Prediction Values and Target Losses", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / "predict_dump_tp_fp_auroc_grid.png", dpi=220)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(
        out_dir / "predict_dump_tp_fp_auroc_metrics.csv", index=False
    )


def main():
    df = _load_predict_dump()
    run_name = datetime.now().strftime("%m-%d-%Y_%H;%M_predict_dump_auroc")
    out_dir = OUTPUT_ROOT / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    _plot_grid(df, out_dir)
    print(f"Saved plot: {out_dir / 'predict_dump_tp_fp_auroc_grid.png'}")
    print(f"Saved metrics: {out_dir / 'predict_dump_tp_fp_auroc_metrics.csv'}")


if __name__ == "__main__":
    main()
