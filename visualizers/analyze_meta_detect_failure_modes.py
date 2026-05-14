import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

META_DETECT_ROOT = r"object_detectors/runs/predict/coco/05-13-2026_22;25_meta_detect"
GT_ROOT = r"object_detectors/runs/predict/coco/05-14-2026_23;11_gt"
META_CLASSIFIER_RUN_ROOT = (
    r"meta_models/meta_classifier/runs/train/coco/05-14-2026_23;17_meta_detect"
)
OUTPUT_ROOT = Path(__file__).resolve().parent / "runs"
THRESHOLD = 0.5
NUM_BINS = 5

KEY_COLUMNS_PRIORITY = [
    ["image_id", "image_path", "raw_pred_idx"],
    ["image_id", "image_path", "pred_idx"],
    ["image_id", "image_path", "xmin", "ymin", "xmax", "ymax"],
]

META_COLUMNS = {
    "image_id",
    "image_path",
    "pred_idx",
    "raw_pred_idx",
    "xmin",
    "ymin",
    "xmax",
    "ymax",
    "score",
    "pred_class",
}

CANDIDATE_FEATURES = [
    "num_candidate_boxes",
    "iou_pb_mean",
    "iou_pb_max",
    "iou_pb_std",
    "score_mean",
    "score_max",
    "score_std",
]

FP_ERROR_TYPES = [
    "localization_fp",
    "classification_fp",
]

FOCUSED_CANDIDATE_FEATURES = {
    "candidate_count": "num_candidate_boxes",
    "candidate_score": "score_max",
}


def resolve_repo_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    return (Path(__file__).resolve().parents[1] / path).resolve()


def resolve_csv(raw_path: str, filename: str) -> Path:
    path = resolve_repo_path(raw_path)
    if path.is_file():
        return path
    csv_path = path / filename
    if csv_path.is_file():
        return csv_path
    raise FileNotFoundError(f"{filename} not found from path: {path}")


def find_eval_data_paths(run_root: Path) -> list[Path]:
    if run_root.is_file():
        return [run_root]
    result_dir = run_root / "results"
    paths = sorted(result_dir.glob("eval_data*.csv")) if result_dir.is_dir() else []
    if not paths:
        paths = sorted(run_root.rglob("eval_data*.csv"))
    if not paths:
        raise FileNotFoundError(f"No eval_data*.csv files found under: {run_root}")
    return paths


def choose_merge_keys(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    for keys in KEY_COLUMNS_PRIORITY:
        if all(key in left.columns and key in right.columns for key in keys):
            return keys
    raise ValueError(
        "Cannot find compatible merge keys. Expected raw_pred_idx, pred_idx, or bbox keys."
    )


def load_eval_dataframe(run_root: Path) -> pd.DataFrame:
    frames = []
    for path in find_eval_data_paths(run_root):
        df = pd.read_csv(path)
        df["eval_source"] = path.name
        frames.append(df)
    eval_df = pd.concat(frames, ignore_index=True)
    required = {"y_test", "y_pred"}
    missing = sorted(required - set(eval_df.columns))
    if missing:
        raise ValueError(f"eval_data is missing required columns: {missing}")
    if not any(set(keys).issubset(eval_df.columns) for keys in KEY_COLUMNS_PRIORITY):
        raise ValueError(
            "eval_data*.csv must include merge keys. Re-run meta classifier after this update "
            "so eval_data contains image_id/image_path and pred_idx or raw_pred_idx."
        )
    return eval_df


def feature_columns(meta_df: pd.DataFrame) -> list[str]:
    cols = []
    for col in meta_df.columns:
        if col in META_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(meta_df[col]):
            cols.append(col)
    return cols


def collapse_feature_groups(columns: list[str]) -> list[dict[str, object]]:
    prob_cols = sorted(
        [col for col in columns if col.startswith("prob_") and col[5:].isdigit()],
        key=lambda col: int(col[5:]),
    )
    prob_set = set(prob_cols)
    groups = []
    inserted_prob = False
    for col in columns:
        if col in prob_set:
            if not inserted_prob:
                groups.append(
                    {"feature": "class_probability_vector", "columns": prob_cols}
                )
                inserted_prob = True
            continue
        groups.append({"feature": col, "columns": [col]})
    return groups


def safe_metrics(y_true, y_score) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if y_true.size < 2 or np.unique(y_true).size < 2:
        return {"auroc": np.nan, "ap": np.nan}
    try:
        auroc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auroc = np.nan
    try:
        ap = float(average_precision_score(y_true, y_score))
    except ValueError:
        ap = np.nan
    return {"auroc": auroc, "ap": ap}


def oriented_scalar_scores(
    values: pd.Series, labels: pd.Series
) -> tuple[np.ndarray, bool, float]:
    scores = pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    metrics = safe_metrics(labels, scores)
    auroc = metrics["auroc"]
    inverted = False
    if np.isfinite(auroc) and auroc < 0.5:
        scores = -scores
        inverted = True
        auroc = 1.0 - auroc
    return scores, inverted, float(auroc) if np.isfinite(auroc) else np.nan


def vector_cv_scores(x: pd.DataFrame, y: pd.Series) -> np.ndarray:
    y_arr = y.astype(int).to_numpy()
    if np.unique(y_arr).size < 2 or len(y_arr) < 10:
        return np.full((len(y_arr),), np.nan, dtype=float)
    min_class = int(np.bincount(y_arr).min())
    n_splits = min(5, min_class)
    if n_splits < 2:
        return np.full((len(y_arr),), np.nan, dtype=float)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs"),
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return cross_val_predict(
        model, x.fillna(0.0).to_numpy(dtype=float), y_arr, cv=cv, method="predict_proba"
    )[:, 1]


def make_output_dir(meta_run: Path, classifier_run: Path) -> Path:
    timestamp = datetime.now().strftime("%m-%d-%Y_%H;%M")
    name = f"{meta_run.name}_{classifier_run.name}"
    safe_name = "".join(
        ch if ch.isalnum() or ch in {"_", "-", ";"} else "_" for ch in name
    ).strip("_")
    return OUTPUT_ROOT / f"{timestamp}_meta_detect_failure_modes_{safe_name}"


def merge_analysis_dataframe(
    eval_df: pd.DataFrame, meta_df: pd.DataFrame, tp_df: pd.DataFrame, threshold: float
) -> pd.DataFrame:
    keys = choose_merge_keys(eval_df, meta_df)
    merged = eval_df.merge(meta_df, on=keys, how="left", suffixes=("", "_meta"))
    tp_keys = choose_merge_keys(merged, tp_df)
    tp_extra_cols = [
        col for col in tp_df.columns if col not in tp_keys and col not in merged.columns
    ]
    merged = merged.merge(
        tp_df[tp_keys + tp_extra_cols], on=tp_keys, how="left", suffixes=("", "_tp")
    )
    if "error_type" not in merged.columns:
        raise ValueError(
            "Merged dataframe is missing error_type. Re-run object detector gt after this update."
        )
    merged["y_test"] = merged["y_test"].astype(int)
    merged["y_pred"] = pd.to_numeric(merged["y_pred"], errors="coerce").fillna(0.0)
    merged["meta_pred_label"] = (merged["y_pred"] >= float(threshold)).astype(int)
    merged["failure_mode"] = "correct"
    merged.loc[
        (merged["y_test"] == 0) & (merged["meta_pred_label"] == 1), "failure_mode"
    ] = "fp_predicted_tp"
    merged.loc[
        (merged["y_test"] == 1) & (merged["meta_pred_label"] == 0), "failure_mode"
    ] = "tp_predicted_fp"
    merged["fp_error_type"] = "tp"
    is_fp = merged["y_test"] == 0
    merged.loc[is_fp, "fp_error_type"] = "localization_fp"
    merged.loc[
        is_fp & (merged["error_type"] == "classification_fp"), "fp_error_type"
    ] = "classification_fp"
    return merged


def analyze_failures(df: pd.DataFrame, out_dir: Path) -> None:
    failures = df[df["failure_mode"] != "correct"].copy()
    failures.to_csv(out_dir / "failure_samples.csv", index=False)

    count_df = (
        df.groupby(["failure_mode", "fp_error_type"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["failure_mode", "count"], ascending=[True, False])
    )
    count_df.to_csv(out_dir / "failure_by_error_type.csv", index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_df = count_df[count_df["failure_mode"] != "correct"]
    if not plot_df.empty:
        pivot = plot_df.pivot_table(
            index="fp_error_type", columns="failure_mode", values="count", fill_value=0
        )
        ax = pivot.plot(kind="bar", figsize=(9, 4.8))
        ax.set_title("MetaDetect Meta Classifier Failures by Error Type")
        ax.set_ylabel("Count")
        ax.set_xlabel("FP error type")
        ax.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(out_dir / "failure_by_error_type.png", dpi=220, bbox_inches="tight")
        plt.close()


def bin_series(values: pd.Series, num_bins: int) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    try:
        return pd.qcut(numeric, q=num_bins, duplicates="drop")
    except ValueError:
        return pd.cut(
            numeric, bins=min(num_bins, max(1, numeric.nunique())), duplicates="drop"
        )


def analyze_candidate_informativeness(df: pd.DataFrame, out_dir: Path) -> None:
    rows = []
    for feature in CANDIDATE_FEATURES:
        if feature not in df.columns:
            continue
        binned = bin_series(df[feature], NUM_BINS)
        for bin_label, subset in df.groupby(binned, observed=True):
            metrics = safe_metrics(subset["y_test"], subset["y_pred"])
            rows.append(
                {
                    "feature": feature,
                    "bin": str(bin_label),
                    "num_samples": int(len(subset)),
                    "num_tp": int(subset["y_test"].sum()),
                    "tp_ratio": (
                        float(subset["y_test"].mean()) if len(subset) else np.nan
                    ),
                    "auroc": metrics["auroc"],
                    "ap": metrics["ap"],
                    "mean_meta_score": (
                        float(subset["y_pred"].mean()) if len(subset) else np.nan
                    ),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "candidate_informativeness_by_bin.csv", index=False)
    if result.empty:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for metric in ["auroc", "ap"]:
        fig, axes = plt.subplots(
            len(CANDIDATE_FEATURES),
            1,
            figsize=(10, max(3.0, 2.2 * len(CANDIDATE_FEATURES))),
        )
        axes = np.atleast_1d(axes)
        for ax, feature in zip(axes, CANDIDATE_FEATURES):
            sub = result[result["feature"] == feature]
            if sub.empty:
                ax.axis("off")
                continue
            ax.plot(range(len(sub)), sub[metric], marker="o", linewidth=1.8)
            ax.set_title(feature)
            ax.set_ylabel(metric.upper())
            ax.set_ylim(0.0, 1.0)
            ax.set_xticks(range(len(sub)))
            ax.set_xticklabels(sub["bin"], rotation=30, ha="right", fontsize=8)
            ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
        axes[-1].set_xlabel("Feature bin")
        fig.tight_layout()
        fig.savefig(
            out_dir / f"candidate_informativeness_{metric}.png",
            dpi=220,
            bbox_inches="tight",
        )
        plt.close(fig)


def compute_binned_performance(
    df: pd.DataFrame, feature: str, scope: str, error_type: str = "all"
) -> pd.DataFrame:
    if feature not in df.columns:
        return pd.DataFrame()
    if scope == "overall":
        subset = df.copy()
    elif scope == "error_type":
        subset = df[
            (df["y_test"] == 1)
            | ((df["y_test"] == 0) & (df["fp_error_type"] == error_type))
        ].copy()
    else:
        raise ValueError(f"Unsupported scope: {scope}")
    if subset.empty:
        return pd.DataFrame()

    rows = []
    binned = bin_series(subset[feature], NUM_BINS)
    for bin_idx, (bin_label, bin_df) in enumerate(
        subset.groupby(binned, observed=True)
    ):
        metrics = safe_metrics(bin_df["y_test"], bin_df["y_pred"])
        rows.append(
            {
                "scope": scope,
                "error_type": error_type,
                "feature": feature,
                "bin_index": int(bin_idx),
                "bin": str(bin_label),
                "num_samples": int(len(bin_df)),
                "num_tp": int(bin_df["y_test"].sum()),
                "num_fp": int((bin_df["y_test"] == 0).sum()),
                "tp_ratio": float(bin_df["y_test"].mean()) if len(bin_df) else np.nan,
                "auroc": metrics["auroc"],
                "ap": metrics["ap"],
                "mean_meta_score": (
                    float(bin_df["y_pred"].mean()) if len(bin_df) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_overall_focused_performance(
    result: pd.DataFrame, feature: str, title: str, out_path: Path
) -> None:
    sub = result[
        (result["scope"] == "overall") & (result["feature"] == feature)
    ].sort_values("bin_index")
    if sub.empty:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(sub))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(x, sub["auroc"], marker="o", linewidth=2.0, label="AUROC")
    ax.plot(x, sub["ap"], marker="s", linewidth=2.0, label="AP")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.set_ylabel("Performance")
    ax.set_xlabel("Bin")
    ax.set_xticks(x)
    ax.set_xticklabels(sub["bin"], rotation=25, ha="right")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend(frameon=False)

    ax2 = ax.twinx()
    ax2.plot(
        x,
        sub["tp_ratio"],
        color="#888888",
        linestyle=":",
        marker="^",
        linewidth=1.5,
        label="TP ratio",
    )
    ax2.set_ylim(0.0, 1.0)
    ax2.set_ylabel("TP ratio")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_error_type_focused_performance(
    result: pd.DataFrame, feature: str, metric: str, title: str, out_path: Path
) -> None:
    sub = result[
        (result["scope"] == "error_type") & (result["feature"] == feature)
    ].copy()
    if sub.empty:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        len(FP_ERROR_TYPES), 1, figsize=(8.8, 3.2 * len(FP_ERROR_TYPES)), sharey=True
    )
    axes = np.atleast_1d(axes)
    for ax, error_type in zip(axes, FP_ERROR_TYPES):
        type_df = sub[sub["error_type"] == error_type].sort_values("bin_index")
        if type_df.empty:
            ax.axis("off")
            continue
        x = np.arange(len(type_df))
        ax.plot(x, type_df[metric], marker="o", linewidth=2.0, label=metric.upper())
        ax.plot(
            x,
            type_df["tp_ratio"],
            color="#888888",
            linestyle=":",
            marker="^",
            linewidth=1.5,
            label="TP ratio",
        )
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"TP vs {error_type}")
        ax.set_ylabel(metric.upper())
        ax.set_xticks(x)
        ax.set_xticklabels(type_df["bin"], rotation=25, ha="right")
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
        ax.legend(frameon=False)
    axes[-1].set_xlabel("Bin")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def analyze_focused_candidate_performance(df: pd.DataFrame, out_dir: Path) -> None:
    frames = []
    for _name, feature in FOCUSED_CANDIDATE_FEATURES.items():
        frames.append(compute_binned_performance(df, feature, scope="overall"))
        for error_type in FP_ERROR_TYPES:
            frames.append(
                compute_binned_performance(
                    df, feature, scope="error_type", error_type=error_type
                )
            )
    result = pd.concat(
        [frame for frame in frames if not frame.empty], ignore_index=True
    )
    result.to_csv(out_dir / "focused_candidate_performance_by_bin.csv", index=False)
    if result.empty:
        return

    plot_overall_focused_performance(
        result,
        FOCUSED_CANDIDATE_FEATURES["candidate_count"],
        "MetaDetect Performance by Number of Candidate Boxes",
        out_dir / "candidate_count_overall_performance.png",
    )
    plot_overall_focused_performance(
        result,
        FOCUSED_CANDIDATE_FEATURES["candidate_score"],
        "MetaDetect Performance by Candidate Score",
        out_dir / "candidate_score_overall_performance.png",
    )
    for metric in ["auroc", "ap"]:
        plot_error_type_focused_performance(
            result,
            FOCUSED_CANDIDATE_FEATURES["candidate_count"],
            metric,
            f"MetaDetect {metric.upper()} by Candidate Count and FP Type",
            out_dir / f"candidate_count_by_error_type_{metric}.png",
        )
        plot_error_type_focused_performance(
            result,
            FOCUSED_CANDIDATE_FEATURES["candidate_score"],
            metric,
            f"MetaDetect {metric.upper()} by Candidate Score and FP Type",
            out_dir / f"candidate_score_by_error_type_{metric}.png",
        )


def analyze_error_type_features(
    df: pd.DataFrame, feature_groups: list[dict[str, object]], out_dir: Path
) -> None:
    rows = []
    for error_type in FP_ERROR_TYPES:
        subset = df[
            (df["y_test"] == 1)
            | ((df["y_test"] == 0) & (df["fp_error_type"] == error_type))
        ].copy()
        if subset["y_test"].nunique() < 2:
            continue
        for group in feature_groups:
            feature_name = str(group["feature"])
            cols = [str(col) for col in group["columns"] if str(col) in subset.columns]
            if not cols:
                continue
            if len(cols) == 1:
                scores, inverted, oriented_auroc = oriented_scalar_scores(
                    subset[cols[0]], subset["y_test"]
                )
                metrics = safe_metrics(subset["y_test"], scores)
                metrics["auroc_oriented"] = oriented_auroc
                metrics["inverted"] = inverted
            else:
                scores = vector_cv_scores(subset[cols], subset["y_test"])
                metrics = safe_metrics(subset["y_test"], scores)
                metrics["auroc_oriented"] = metrics["auroc"]
                metrics["inverted"] = False
            rows.append(
                {
                    "error_type": error_type,
                    "feature": feature_name,
                    "num_columns": int(len(cols)),
                    "num_samples": int(len(subset)),
                    "num_tp": int(subset["y_test"].sum()),
                    "num_fp": int((subset["y_test"] == 0).sum()),
                    "auroc": metrics["auroc"],
                    "auroc_oriented": metrics["auroc_oriented"],
                    "ap": metrics["ap"],
                    "inverted": bool(metrics["inverted"]),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "single_feature_by_error_type.csv", index=False)
    if result.empty:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for error_type in FP_ERROR_TYPES:
        sub = (
            result[result["error_type"] == error_type]
            .sort_values("auroc_oriented", ascending=False)
            .head(10)
        )
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.barh(sub["feature"][::-1], sub["auroc_oriented"][::-1])
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("Single-feature AUROC (oriented)")
        ax.set_title(f"Top MetaDetect Features: TP vs {error_type}")
        ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.4)
        fig.tight_layout()
        fig.savefig(
            out_dir / f"top_features_{error_type}.png", dpi=220, bbox_inches="tight"
        )
        plt.close(fig)

    pivot = result.pivot_table(
        index="error_type", columns="feature", values="auroc_oriented", aggfunc="mean"
    )
    top_features = (
        result.groupby("feature")["auroc_oriented"]
        .max()
        .sort_values(ascending=False)
        .head(25)
        .index.tolist()
    )
    if top_features:
        heat = pivot[top_features]
        fig, ax = plt.subplots(figsize=(max(10, 0.35 * len(top_features)), 4.2))
        image = ax.imshow(
            heat.to_numpy(dtype=float),
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
            aspect="auto",
        )
        ax.set_xticks(range(len(top_features)))
        ax.set_xticklabels(top_features, rotation=90, fontsize=8)
        ax.set_yticks(range(len(heat.index)))
        ax.set_yticklabels(heat.index)
        ax.set_title("Error Type Specific Single-Feature AUROC")
        fig.colorbar(image, ax=ax, label="AUROC")
        fig.tight_layout()
        fig.savefig(
            out_dir / "single_feature_by_error_type_heatmap.png",
            dpi=220,
            bbox_inches="tight",
        )
        plt.close(fig)


def main() -> None:
    meta_csv = resolve_csv(META_DETECT_ROOT, "meta_detect.csv")
    gt_csv = resolve_csv(GT_ROOT, "tp.csv")
    classifier_root = resolve_repo_path(META_CLASSIFIER_RUN_ROOT)
    eval_df = load_eval_dataframe(classifier_root)
    meta_df = pd.read_csv(meta_csv)
    tp_df = pd.read_csv(gt_csv)

    out_dir = make_output_dir(meta_csv.parent, classifier_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = merge_analysis_dataframe(eval_df, meta_df, tp_df, threshold=float(THRESHOLD))
    df.to_csv(out_dir / "merged_analysis_dataframe.csv", index=False)
    features = collapse_feature_groups(feature_columns(meta_df))

    analyze_failures(df, out_dir)
    analyze_candidate_informativeness(df, out_dir)
    analyze_focused_candidate_performance(df, out_dir)
    analyze_error_type_features(df, features, out_dir)

    metadata = {
        "meta_detect_csv": str(meta_csv),
        "gt_csv": str(gt_csv),
        "meta_classifier_run_root": str(classifier_root),
        "threshold": float(THRESHOLD),
        "num_eval_rows": int(len(eval_df)),
        "num_merged_rows": int(len(df)),
        "num_feature_groups": int(len(features)),
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Saved MetaDetect failure mode analysis: {out_dir}")


if __name__ == "__main__":
    main()
