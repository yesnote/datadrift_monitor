from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KEYS = ["image_id", "image_path", "raw_pred_idx"]
BASE_GT_COLUMNS = KEYS + ["tp", "gt_iou", "max_iou"]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "documents/appendix/generated"


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    display_name: str
    summary_name: str
    gt_root: Path
    unto_o_root: Path
    grad_grid_root: Path


DATASET_CONFIGS: dict[str, DatasetConfig] = {
    "coco": DatasetConfig(
        key="coco",
        display_name="MS COCO",
        summary_name="MS COCO train2017",
        gt_root=PROJECT_ROOT / "object_detectors/runs/yolov5/predict/coco/06-15-2026_18;54_gt",
        unto_o_root=PROJECT_ROOT / "object_detectors/runs/yolov5/predict/coco/06-16-2026_04;47_null_detect",
        grad_grid_root=PROJECT_ROOT / "object_detectors/runs/yolov5/predict/coco/06-16-2026_06;46_layer_grad_grid",
    ),
    "voc": DatasetConfig(
        key="voc",
        display_name="Pascal VOC",
        summary_name="Pascal VOC 2012 train",
        gt_root=PROJECT_ROOT / "object_detectors/runs/yolov5/predict/voc/06-14-2026_13;36_gt",
        unto_o_root=PROJECT_ROOT / "object_detectors/runs/yolov5/predict/voc/06-14-2026_15;54_null_detect",
        grad_grid_root=PROJECT_ROOT / "object_detectors/runs/yolov5/predict/voc/06-14-2026_17;09_layer_grad_grid",
    ),
}

GRAD_COMPONENTS = {
    "Localization gradient only": {
        "suffix": "t-null__term-bbox__b-box_l1-pred",
        "prefix": "bbox_loss_",
    },
    "Classification gradient only": {
        "suffix": "t-null__term-cls__c-kl-pred",
        "prefix": "cls_loss_",
    },
    "Detection-score gradient only": {
        "suffix": "t-null__term-obj__o-bce-pred",
        "prefix": "obj_loss_",
    },
}

GRAD_LAYER_ROWS = [
    ("YOLOv5", "All components", "model.24.m.0, model.24.m.1, model.24.m.2"),
    ("FCOS", "Localization", "detector_model.rpn.head.bbox_pred"),
    ("FCOS", "Classification", "detector_model.rpn.head.cls_logits"),
    ("FCOS", "Detection score", "detector_model.rpn.head.centerness"),
    ("Faster R-CNN", "RPN objectness", "rpn.head.conv, rpn.head.cls_logits"),
    ("Faster R-CNN", "RPN box", "rpn.head.conv, rpn.head.bbox_pred"),
    ("Faster R-CNN", "RoI class", "roi_heads.box_head.fc7, roi_heads.box_predictor.cls_score"),
    ("Faster R-CNN", "RoI box", "roi_heads.box_head.fc7, roi_heads.box_predictor.bbox_pred"),
]


@dataclass
class EvalResult:
    dataset: str
    name: str
    num_rows: int
    num_features: int
    tp_ratio: float
    auroc_mean: float
    auroc_std: float
    ap_mean: float
    ap_std: float
    fpr95_mean: float
    fpr95_std: float
    r2_mean: float
    r2_std: float


def resolve_path(raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def read_header(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def read_csv(path: Path, usecols: list[str], limit_rows: int | None) -> pd.DataFrame:
    return pd.read_csv(path, usecols=list(dict.fromkeys(usecols)), nrows=limit_rows)


def load_gt(gt_root: Path, limit_rows: int | None) -> pd.DataFrame:
    gt_csv = gt_root / "tp.csv"
    header = read_header(gt_csv)
    cols = [c for c in BASE_GT_COLUMNS if c in header]
    missing = set(KEYS + ["tp"]) - set(cols)
    if missing:
        raise ValueError(f"Missing required columns in {gt_csv}: {sorted(missing)}")
    gt = read_csv(gt_csv, cols, limit_rows)
    if "gt_iou" not in gt.columns:
        if "max_iou" not in gt.columns:
            raise ValueError("tp.csv must contain gt_iou or max_iou for regression.")
        gt["gt_iou"] = gt["max_iou"]
    return gt


def merge_features(gt: pd.DataFrame, feature_csv: Path, feature_cols: list[str], limit_rows: int | None) -> pd.DataFrame:
    missing = set(KEYS + feature_cols) - set(read_header(feature_csv))
    if missing:
        raise ValueError(f"Missing columns in {feature_csv}: {sorted(missing)[:20]}")
    feature_df = read_csv(feature_csv, KEYS + feature_cols, limit_rows)
    return gt.merge(feature_df, on=KEYS, how="inner", validate="one_to_one")


def fpr_at_tpr(y_true: np.ndarray, y_score: np.ndarray, target_tpr: float = 0.95) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    mask = tpr >= target_tpr
    if not np.any(mask):
        return 1.0
    return float(np.min(fpr[mask]))


def sanitize_x(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    return np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)


def classifier_setting(classifier: str) -> str:
    if classifier == "xgb":
        return "StandardScaler + XGBClassifier(eval_metric=logloss, random_state=42)"
    if classifier == "sklearn-gb":
        return "StandardScaler + GradientBoostingClassifier(random_state=42; dry-run fallback)"
    raise ValueError(f"Unsupported classifier backend: {classifier}")


def classifier_estimator(random_seed: int, n_jobs: int, classifier: str) -> Pipeline:
    if classifier == "xgb":
        if xgb is None:
            raise ImportError(
                "xgboost is required for the default classifier ablation. "
                "Install xgboost or pass --classifier sklearn-gb for a lightweight dry run."
            )
        clf = xgb.XGBClassifier(eval_metric="logloss", random_state=random_seed, n_jobs=n_jobs, verbosity=0)
    elif classifier == "sklearn-gb":
        clf = GradientBoostingClassifier(random_state=random_seed)
    else:
        raise ValueError(f"Unsupported classifier backend: {classifier}")
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def regressor_estimator(random_seed: int) -> Pipeline:
    reg = GradientBoostingRegressor(random_state=random_seed)
    return Pipeline([("scaler", StandardScaler()), ("reg", reg)])


def eval_classifier(x: np.ndarray, y: np.ndarray, repeats: int, split: float, random_seed: int, n_jobs: int, classifier: str) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    indices = np.arange(len(y))
    for i in range(repeats):
        train_idx, test_idx = train_test_split(
            indices,
            test_size=split,
            random_state=random_seed + i,
            stratify=y,
            shuffle=True,
        )
        model = classifier_estimator(random_seed, n_jobs, classifier)
        model.fit(x[train_idx], y[train_idx])
        y_pred = model.predict_proba(x[test_idx])[:, 1]
        rows.append(
            {
                "split_index": i,
                "auroc": float(roc_auc_score(y[test_idx], y_pred)),
                "ap": float(average_precision_score(y[test_idx], y_pred)),
                "fpr95": float(fpr_at_tpr(y[test_idx], y_pred)),
            }
        )
    return pd.DataFrame(rows)


def eval_regressor(x: np.ndarray, y: np.ndarray, repeats: int, split: float, random_seed: int) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    indices = np.arange(len(y))
    for i in range(repeats):
        train_idx, test_idx = train_test_split(
            indices,
            test_size=split,
            random_state=random_seed + i,
            shuffle=True,
        )
        model = regressor_estimator(random_seed)
        model.fit(x[train_idx], y[train_idx])
        y_pred = np.clip(model.predict(x[test_idx]), 0.0, 1.0)
        rows.append({"split_index": i, "r2": float(r2_score(y[test_idx], y_pred))})
    return pd.DataFrame(rows)


def summarize_result(dataset: str, name: str, df: pd.DataFrame, feature_cols: list[str], cls_df: pd.DataFrame, reg_df: pd.DataFrame) -> EvalResult:
    y = df["tp"].astype(int).to_numpy()
    return EvalResult(
        dataset=dataset,
        name=name,
        num_rows=int(len(df)),
        num_features=int(len(feature_cols)),
        tp_ratio=float(np.mean(y)) if len(y) else 0.0,
        auroc_mean=float(cls_df["auroc"].mean()),
        auroc_std=float(cls_df["auroc"].std(ddof=1)) if len(cls_df) > 1 else 0.0,
        ap_mean=float(cls_df["ap"].mean()),
        ap_std=float(cls_df["ap"].std(ddof=1)) if len(cls_df) > 1 else 0.0,
        fpr95_mean=float(cls_df["fpr95"].mean()),
        fpr95_std=float(cls_df["fpr95"].std(ddof=1)) if len(cls_df) > 1 else 0.0,
        r2_mean=float(reg_df["r2"].mean()),
        r2_std=float(reg_df["r2"].std(ddof=1)) if len(reg_df) > 1 else 0.0,
    )


def evaluate_group(dataset: str, name: str, df: pd.DataFrame, feature_cols: list[str], repeats: int, split: float, random_seed: int, n_jobs: int, classifier: str) -> EvalResult:
    x = sanitize_x(df[feature_cols].to_numpy(dtype=np.float32, copy=False))
    y_cls = df["tp"].astype(int).to_numpy()
    y_reg = df["gt_iou"].astype(float).to_numpy()
    cls_df = eval_classifier(x, y_cls, repeats, split, random_seed, n_jobs, classifier)
    reg_df = eval_regressor(x, y_reg, repeats, split, random_seed)
    return summarize_result(dataset, name, df, feature_cols, cls_df, reg_df)


def sorted_probability_columns(header: list[str]) -> list[str]:
    columns = []
    for col in header:
        match = re.fullmatch(r"prob_(\d+)", col)
        if match:
            columns.append((int(match.group(1)), col))
    return [col for _, col in sorted(columns)]


def unto_o_feature_sets(header: list[str]) -> dict[str, list[str]]:
    prob_cols = sorted_probability_columns(header)
    final_class = ["prob_sum"] + prob_cols if "prob_sum" in header else prob_cols
    final_score = ["final_score"] if "final_score" in header else []
    final_loc = [c for c in ["size", "circum", "size_circum"] if c in header]
    target_class = ["cls_loss"] if "cls_loss" in header else []
    target_score = ["obj_loss"] if "obj_loss" in header else []
    target_loc = [c for c in ["x_loss", "y_loss", "w_loss", "h_loss", "size_diff", "circum_diff", "size_circum_diff"] if c in header]
    return {
        "Full UnTO-O": final_class + final_score + final_loc + target_class + target_score + target_loc,
        "Final-detection only": final_class + final_score + final_loc,
        "Target-relative only": target_class + target_score + target_loc,
        "Final class+score": final_class + final_score,
        "Final localization": final_loc,
        "Target class": target_class,
        "Target score": target_score,
        "Target localization": target_loc,
    }


def load_unto_o(gt: pd.DataFrame, unto_o_root: Path, limit_rows: int | None) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    feature_csv = unto_o_root / "null_detect.csv"
    feature_sets = unto_o_feature_sets(read_header(feature_csv))
    all_cols: list[str] = []
    for cols in feature_sets.values():
        all_cols.extend(cols)
    df = merge_features(gt, feature_csv, list(dict.fromkeys(all_cols)), limit_rows)
    return df, feature_sets


def find_component_csv(grid_root: Path, suffix: str) -> Path:
    matches = sorted(path for path in grid_root.iterdir() if path.is_dir() and path.name.endswith(suffix))
    if not matches:
        raise FileNotFoundError(f"No gradient component directory ending with {suffix!r} under {grid_root}")
    if len(matches) > 1:
        raise ValueError(f"Multiple gradient component directories ending with {suffix!r} under {grid_root}: {[p.name for p in matches]}")
    csv_path = matches[0] / "layer_grad.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    return csv_path


def load_gradient_components(gt: pd.DataFrame, grid_root: Path, limit_rows: int | None) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    frames: list[pd.DataFrame] = [gt.set_index(KEYS)]
    groups: dict[str, list[str]] = {}
    for name, spec in GRAD_COMPONENTS.items():
        csv_path = find_component_csv(grid_root, spec["suffix"])
        header = read_header(csv_path)
        cols = [c for c in header if c.startswith(spec["prefix"])]
        if not cols:
            raise ValueError(f"No gradient columns matching {spec['prefix']} in {csv_path}")
        df = read_csv(csv_path, KEYS + cols, limit_rows).set_index(KEYS)
        frames.append(df)
        groups[name] = cols
    merged = pd.concat(frames, axis=1, join="inner").reset_index()
    all_cols: list[str] = []
    for cols in groups.values():
        all_cols.extend(cols)
    groups["All components"] = all_cols
    return merged, groups


def result_to_dict(result: EvalResult) -> dict[str, float | int | str]:
    return {
        "dataset": result.dataset,
        "method": result.name,
        "num_rows": result.num_rows,
        "num_features": result.num_features,
        "tp_ratio": result.tp_ratio,
        "auroc_mean": result.auroc_mean,
        "auroc_std": result.auroc_std,
        "ap_mean": result.ap_mean,
        "ap_std": result.ap_std,
        "fpr95_mean": result.fpr95_mean,
        "fpr95_std": result.fpr95_std,
        "r2_mean": result.r2_mean,
        "r2_std": result.r2_std,
    }


def fmt_metric(value: float, std: float, percent: bool = True) -> str:
    if math.isnan(value):
        return "--"
    if percent:
        return f"{value * 100:.2f}\\tabstd{{{std * 100:.2f}}}"
    return f"{value:.2f}\\tabstd{{{std:.3f}}}"


def escape_tex(text: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def write_ablation_table(path: Path, label: str, caption: str, results: list[EvalResult]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{L{0.13\linewidth}L{0.24\linewidth}ccccc}",
        r"\toprule",
        r"Dataset & Variant & \# Feat. & AUROC & AP & FPR95 & $R^2$ \\",
        r"\midrule",
    ]
    previous_dataset = None
    for r in results:
        dataset = escape_tex(r.dataset) if r.dataset != previous_dataset else ""
        previous_dataset = r.dataset
        lines.append(
            f"{dataset} & {escape_tex(r.name)} & {r.num_features} & "
            f"{fmt_metric(r.auroc_mean, r.auroc_std)} & "
            f"{fmt_metric(r.ap_mean, r.ap_std)} & "
            f"{fmt_metric(r.fpr95_mean, r.fpr95_std)} & "
            f"{fmt_metric(r.r2_mean, r.r2_std, percent=False)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_gradient_layers_table(path: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Gradient extraction layers used for UnTO-G.}",
        r"\label{tab:app_gradient_layers}",
        r"\begin{tabular}{L{0.18\linewidth}L{0.28\linewidth}L{0.44\linewidth}}",
        r"\toprule",
        r"Detector & Component & Layers \\",
        r"\midrule",
    ]
    for detector, component, layers in GRAD_LAYER_ROWS:
        lines.append(f"{escape_tex(detector)} & {escape_tex(component)} & {escape_tex(layers)} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_hyperparameters(path: Path, repeats: int, split: float, random_seed: int, classifier: str) -> None:
    clf_params = classifier_setting(classifier)
    reg_params = "StandardScaler + GradientBoostingRegressor(random_state=42)"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Meta-model and split settings used for appendix ablations.}",
        r"\label{tab:app_hyperparameters}",
        r"\begin{tabular}{L{0.28\linewidth}L{0.62\linewidth}}",
        r"\toprule",
        r"Item & Setting \\",
        r"\midrule",
        f"Classifier & {escape_tex(clf_params)} " + r"\\",
        f"Regressor & {escape_tex(reg_params)} " + r"\\",
        f"Splits & {repeats} repeated train/test splits " + r"\\",
        f"Test ratio & {split:.2f} " + r"\\",
        f"Split seeds & {random_seed}+i for split index i " + r"\\",
        r"Hyperparameter search & Disabled \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dataset_summary_row(config: DatasetConfig, gt: pd.DataFrame, limit_rows: int | None) -> dict[str, float | int | str]:
    return {
        "detector": "YOLOv5",
        "dataset": config.summary_name,
        "display_name": config.display_name,
        "detections": int(len(gt)),
        "tp_ratio": float(gt["tp"].astype(int).mean()) if len(gt) else 0.0,
        "regression_target": "gt_iou",
        "limit_rows": limit_rows or "",
    }


def write_raw_dataset_summary(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def write_dataset_summary(path: Path, rows: list[dict[str, float | int | str]], limit_rows: int | None) -> None:
    caption_suffix = " on the loaded evaluation rows"
    if limit_rows is not None:
        caption_suffix += f" (limited to {limit_rows:,} rows for a dry run)"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{YOLOv5 appendix ablation data summary{caption_suffix}.}}",
        r"\label{tab:app_dataset_summary}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Dataset & Detections & TP Ratio & Regression Target \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{escape_tex(str(row['display_name']))} & {int(row['detections']):,} & "
            f"{float(row['tp_ratio']) * 100:.2f}\\% & best same-class IoU " + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_raw_hyperparameters_csv(path: Path, repeats: int, split: float, random_seed: int, classifier: str) -> None:
    rows = [
        {"item": "classifier", "setting": classifier_setting(classifier)},
        {"item": "regressor", "setting": "StandardScaler + GradientBoostingRegressor(random_state=42)"},
        {"item": "repeats", "setting": repeats},
        {"item": "test_split", "setting": split},
        {"item": "split_seed", "setting": f"{random_seed}+i"},
        {"item": "hyperparameter_search", "setting": False},
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def parse_dataset_names(raw: str) -> list[str]:
    names = [name.strip().lower() for name in raw.split(",") if name.strip()]
    if not names:
        raise ValueError("At least one dataset must be selected.")
    invalid = [name for name in names if name not in DATASET_CONFIGS and name != "custom"]
    if invalid:
        raise ValueError(f"Unknown dataset names: {invalid}. Available: {sorted(DATASET_CONFIGS)} or custom")
    if "custom" in names and len(names) > 1:
        raise ValueError("The custom dataset cannot be combined with named datasets.")
    return names


def selected_configs(args: argparse.Namespace) -> list[DatasetConfig]:
    root_args = [args.gt_root, args.unto_o_root, args.grad_grid_root]
    if any(root_args):
        if not all(root_args):
            raise ValueError("--gt-root, --unto-o-root, and --grad-grid-root must be provided together for a custom dataset.")
        return [
            DatasetConfig(
                key="custom",
                display_name=args.custom_dataset_name,
                summary_name=args.custom_dataset_name,
                gt_root=resolve_path(args.gt_root),
                unto_o_root=resolve_path(args.unto_o_root),
                grad_grid_root=resolve_path(args.grad_grid_root),
            )
        ]
    names = parse_dataset_names(args.datasets)
    if names == ["custom"]:
        raise ValueError("--datasets custom requires --gt-root, --unto-o-root, and --grad-grid-root.")
    return [DATASET_CONFIGS[name] for name in names]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build appendix ablation results for UnTO.")
    parser.add_argument("--datasets", default="coco,voc", help="Comma-separated datasets to evaluate: coco, voc, or custom.")
    parser.add_argument("--gt-root", default=None, help="Custom GT run root containing tp.csv.")
    parser.add_argument("--unto-o-root", default=None, help="Custom UnTO-O run root containing null_detect.csv.")
    parser.add_argument("--grad-grid-root", default=None, help="Custom UnTO-G grid root containing component layer_grad.csv files.")
    parser.add_argument("--custom-dataset-name", default="Custom", help="Display name used when custom roots are provided.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--limit-rows", type=int, default=None, help="Limit rows for a quick dry run.")
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--split", type=float, default=0.3)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--classifier", choices=["xgb", "sklearn-gb"], default="xgb", help="Classifier backend; xgb is the main-paper setting, sklearn-gb is for dry-run validation when xgboost is unavailable.")
    parser.add_argument("--skip-unto-o", action="store_true")
    parser.add_argument("--skip-unto-g", action="store_true")
    return parser.parse_args()


def evaluate_unto_o(config: DatasetConfig, gt: pd.DataFrame, args: argparse.Namespace) -> list[EvalResult]:
    unto_o_df, feature_sets = load_unto_o(gt, config.unto_o_root, args.limit_rows)
    results: list[EvalResult] = []
    for name, cols in feature_sets.items():
        print(f"[{config.display_name}][UnTO-O] {name}: {len(cols)} features, {len(unto_o_df)} rows")
        results.append(evaluate_group(config.display_name, name, unto_o_df, cols, args.repeats, args.split, args.random_seed, args.n_jobs, args.classifier))
    return results


def evaluate_unto_g(config: DatasetConfig, gt: pd.DataFrame, args: argparse.Namespace) -> list[EvalResult]:
    grad_df, grad_groups = load_gradient_components(gt, config.grad_grid_root, args.limit_rows)
    order = ["Localization gradient only", "Classification gradient only", "Detection-score gradient only", "All components"]
    results: list[EvalResult] = []
    for name in order:
        cols = grad_groups[name]
        print(f"[{config.display_name}][UnTO-G] {name}: {len(cols)} features, {len(grad_df)} rows")
        results.append(evaluate_group(config.display_name, name, grad_df, cols, args.repeats, args.split, args.random_seed, args.n_jobs, args.classifier))
    return results


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configs = selected_configs(args)

    write_raw_hyperparameters_csv(output_dir / "meta_model_hyperparameters.csv", args.repeats, args.split, args.random_seed, args.classifier)
    write_hyperparameters(output_dir / "hyperparameters_table.tex", args.repeats, args.split, args.random_seed, args.classifier)
    write_gradient_layers_table(output_dir / "gradient_layers_table.tex")

    dataset_rows: list[dict[str, float | int | str]] = []
    unto_o_results: list[EvalResult] = []
    unto_g_results: list[EvalResult] = []
    metadata = {
        "selected_datasets": [config.key for config in configs],
        "limit_rows": args.limit_rows,
        "repeats": args.repeats,
        "split": args.split,
        "random_seed": args.random_seed,
        "classifier": args.classifier,
        "datasets": {},
    }

    for config in configs:
        print(f"[Dataset] {config.display_name}")
        gt = load_gt(config.gt_root, args.limit_rows)
        dataset_rows.append(dataset_summary_row(config, gt, args.limit_rows))
        metadata["datasets"][config.key] = {
            "display_name": config.display_name,
            "gt_root": str(config.gt_root),
            "unto_o_root": str(config.unto_o_root),
            "grad_grid_root": str(config.grad_grid_root),
            "loaded_rows": int(len(gt)),
        }
        if not args.skip_unto_o:
            unto_o_results.extend(evaluate_unto_o(config, gt, args))
        if not args.skip_unto_g:
            unto_g_results.extend(evaluate_unto_g(config, gt, args))

    write_raw_dataset_summary(output_dir / "dataset_summary.csv", dataset_rows)
    write_dataset_summary(output_dir / "dataset_summary_table.tex", dataset_rows, args.limit_rows)
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if unto_o_results:
        pd.DataFrame([result_to_dict(r) for r in unto_o_results]).to_csv(output_dir / "unto_o_ablation_results.csv", index=False)
        write_ablation_table(
            output_dir / "unto_o_ablation_table.tex",
            "tab:app_unto_o_ablation",
            "UnTO-O indicator ablation on YOLOv5 using MS COCO and Pascal VOC.",
            unto_o_results,
        )
    if unto_g_results:
        pd.DataFrame([result_to_dict(r) for r in unto_g_results]).to_csv(output_dir / "unto_g_component_ablation_results.csv", index=False)
        write_ablation_table(
            output_dir / "unto_g_component_table.tex",
            "tab:app_unto_g_component_ablation",
            "UnTO-G component-gradient ablation on YOLOv5 using MS COCO and Pascal VOC.",
            unto_g_results,
        )

    print(f"Appendix outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
