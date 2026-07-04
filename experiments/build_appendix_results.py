from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, train_test_split

from meta_models.commands.common import (
    build_feature_matrix,
    infer_feature_spec,
    load_training_dataframe,
    sanitize_feature_matrix,
)
from meta_models.losses.meta_classifier import (
    compute_ace,
    compute_ece,
    compute_fpr_at_tpr,
    evaluate_classifier,
)
from meta_models.losses.meta_regressor import evaluate_regressor
from meta_models.models.meta_classifier import (
    build_estimator as build_classifier_estimator,
    param_grid as classifier_param_grid,
)
from meta_models.models.meta_regressor import (
    build_estimator as build_regressor_estimator,
    param_grid as regressor_param_grid,
)

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
except Exception:  # pragma: no cover
    SMOTE = None
    ImbPipeline = None


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "documents/appendix/generated"
DEFAULT_CLASSIFIER_CONFIG = PROJECT_ROOT / "meta_models/configs/meta_classifier/train.yaml"
DEFAULT_REGRESSOR_CONFIG = PROJECT_ROOT / "meta_models/configs/meta_regressor/train.yaml"


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
    "Localization gradient only": "t-null__term-bbox__b-box_l1-pred",
    "Classification gradient only": "t-null__term-cls__c-kl-pred",
    "Detection-score gradient only": "t-null__term-obj__o-bce-pred",
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
    ece_mean: float
    ece_std: float
    ace_mean: float
    ace_std: float


@dataclass
class LoadedTaskData:
    df: pd.DataFrame
    label_col: str
    features: list[str]
    root_info: dict[str, Any]
    x: np.ndarray
    y: np.ndarray
    nonfinite_stats: dict[str, int]


def resolve_path(raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def path_for_config(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must contain a mapping: {path}")
    return data


def dataset_config_from_base(base_config: dict[str, Any], input_root: Path | list[Path], gt_root: Path) -> dict[str, Any]:
    cfg = copy.deepcopy(base_config)
    roots = input_root if isinstance(input_root, list) else [input_root]
    cfg.setdefault("dataset", {})["input_root"] = [path_for_config(root) for root in roots]
    cfg["dataset"]["gt_root"] = path_for_config(gt_root)
    return cfg


def apply_limit(df: pd.DataFrame, limit_rows: int | None) -> pd.DataFrame:
    if limit_rows is None:
        return df
    return df.head(limit_rows).copy()


def prepare_loaded_data(dataset_cfg: dict[str, Any], task: str, limit_rows: int | None) -> LoadedTaskData:
    df, label_col, features, root_info = load_training_dataframe(dataset_cfg["dataset"], task=task)
    df = apply_limit(df, limit_rows)
    spec = infer_feature_spec(df, features)
    x = build_feature_matrix(df, spec)
    x, nonfinite_stats = sanitize_feature_matrix(x)
    if task == "classifier":
        y = df[label_col].astype(int).to_numpy()
    else:
        y = df[label_col].astype(float).to_numpy()
    return LoadedTaskData(
        df=df,
        label_col=label_col,
        features=features,
        root_info=root_info,
        x=x,
        y=y,
        nonfinite_stats=nonfinite_stats,
    )


def make_subset_data(data: LoadedTaskData, feature_columns: list[str]) -> LoadedTaskData:
    missing = [col for col in feature_columns if col not in data.features]
    if missing:
        raise ValueError(f"Requested feature columns are not loaded: {missing[:10]}")
    spec = infer_feature_spec(data.df, feature_columns)
    x = build_feature_matrix(data.df, spec)
    x, nonfinite_stats = sanitize_feature_matrix(x)
    return LoadedTaskData(
        df=data.df,
        label_col=data.label_col,
        features=feature_columns,
        root_info=data.root_info,
        x=x,
        y=data.y,
        nonfinite_stats=nonfinite_stats,
    )


def classifier_metric_row(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    auroc, ap = evaluate_classifier(y_true, y_score)
    return {
        "auroc": float(auroc),
        "ap": float(ap),
        "fpr95": float(compute_fpr_at_tpr(y_true, y_score)),
        "ece": float(compute_ece(y_true, y_score)),
        "ace": float(compute_ace(y_true, y_score)),
    }


def apply_classifier_augmentation(
    x: np.ndarray,
    y: np.ndarray,
    augmentation: str,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if augmentation == "none":
        return x, y
    if augmentation == "smote":
        if SMOTE is None:
            raise ImportError("imblearn is required for augmentation='smote'.")
        sampler = SMOTE(random_state=int(random_seed))
        return sampler.fit_resample(x, y)
    raise ValueError(f"Unsupported augmentation: {augmentation}")


def strip_search_params(params: dict[str, Any], strip_prefix: bool) -> dict[str, Any]:
    if not strip_prefix:
        return dict(params)
    prefix = "estimator__"
    return {key[len(prefix):] if key.startswith(prefix) else key: value for key, value in params.items()}


def build_classifier_with_config(config: dict[str, Any], x: np.ndarray, y: np.ndarray):
    model_cfg = config["model"]
    exp_cfg = config["experiment"]
    model_name = str(model_cfg.get("type", "gb_classifier"))
    device = str(model_cfg.get("device", "cpu"))
    random_seed = int(model_cfg.get("random_seed", 42))
    estimator = build_classifier_estimator(model_name, device=device, random_seed=random_seed)

    if not bool(model_cfg.get("search", False)):
        return estimator

    augmentation = str(exp_cfg.get("augmentation", "none"))
    grid = classifier_param_grid(model_name)
    search_estimator = estimator
    strip_prefix = False
    if augmentation == "smote":
        if SMOTE is None or ImbPipeline is None:
            raise ImportError("imblearn is required for augmentation='smote'.")
        search_estimator = ImbPipeline(
            [("smote", SMOTE(random_state=random_seed)), ("estimator", estimator)]
        )
        grid = {f"estimator__{key}": value for key, value in grid.items()}
        strip_prefix = True
    elif augmentation != "none":
        raise ValueError(f"Unsupported augmentation: {augmentation}")

    search = GridSearchCV(
        estimator=search_estimator,
        param_grid=grid,
        scoring=str(model_cfg.get("search_scoring", "roc_auc")),
        n_jobs=int(exp_cfg.get("n_jobs", 8)),
        cv=5,
        verbose=1,
    )
    search.fit(x, y)
    estimator.set_params(**strip_search_params(dict(search.best_params_), strip_prefix))
    return estimator


def build_regressor_with_config(config: dict[str, Any], x: np.ndarray, y: np.ndarray):
    model_cfg = config["model"]
    exp_cfg = config["experiment"]
    model_name = str(model_cfg.get("type", "gb_regressor"))
    device = str(model_cfg.get("device", "cpu"))
    random_seed = int(model_cfg.get("random_seed", 42))
    estimator = build_regressor_estimator(model_name, device=device, random_seed=random_seed)

    if not bool(model_cfg.get("search", False)):
        return estimator

    search = GridSearchCV(
        estimator=estimator,
        param_grid=regressor_param_grid(model_name),
        scoring=str(model_cfg.get("search_scoring", "neg_mean_absolute_error")),
        n_jobs=int(exp_cfg.get("n_jobs", 8)),
        cv=5,
        verbose=1,
    )
    search.fit(x, y)
    estimator.set_params(**dict(search.best_params_))
    return estimator


def evaluate_classifier_protocol(data: LoadedTaskData, config: dict[str, Any]) -> pd.DataFrame:
    model_cfg = config["model"]
    exp_cfg = config["experiment"]
    random_seed = int(model_cfg.get("random_seed", 42))
    augmentation = str(exp_cfg.get("augmentation", "none"))
    process = str(exp_cfg.get("process", "kfold")).strip().lower()
    x, y = data.x, data.y.astype(int)
    rows: list[dict[str, Any]] = []

    if process == "repeat":
        repeat_cfg = exp_cfg.get("repeat", {})
        split = float(repeat_cfg.get("split", 0.3))
        repeats = int(repeat_cfg.get("repeats", 15))
        indices = np.arange(len(y))
        for i in range(repeats):
            train_idx, test_idx = train_test_split(
                indices,
                test_size=split,
                random_state=random_seed + i,
                stratify=y,
                shuffle=True,
            )
            x_train, x_test = x[train_idx], x[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            x_train, y_train = apply_classifier_augmentation(
                x_train,
                y_train,
                augmentation,
                random_seed=random_seed + i,
            )
            estimator = build_classifier_with_config(config, x_train, y_train)
            estimator.fit(x_train, y_train)
            y_pred = estimator.predict_proba(x_test)[:, 1]
            rows.append({"row_type": "split", "split_index": int(i), **classifier_metric_row(y_test, y_pred)})
        return pd.DataFrame(rows)

    if process == "kfold":
        kfold_cfg = exp_cfg.get("kfold", {})
        num_fold = int(kfold_cfg.get("num_fold", 10))
        kfold = StratifiedKFold(n_splits=num_fold, shuffle=True, random_state=random_seed)
        for i, (train_idx, test_idx) in enumerate(kfold.split(x, y)):
            x_train, x_test = x[train_idx], x[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            x_train, y_train = apply_classifier_augmentation(
                x_train,
                y_train,
                augmentation,
                random_seed=random_seed + i,
            )
            estimator = build_classifier_with_config(config, x_train, y_train)
            estimator.fit(x_train, y_train)
            y_pred = estimator.predict_proba(x_test)[:, 1]
            rows.append({"row_type": "split", "split_index": int(i), **classifier_metric_row(y_test, y_pred)})
        return pd.DataFrame(rows)

    raise ValueError("experiment.process must be 'kfold' or 'repeat'.")


def evaluate_regressor_protocol(data: LoadedTaskData, config: dict[str, Any]) -> pd.DataFrame:
    model_cfg = config["model"]
    exp_cfg = config["experiment"]
    random_seed = int(model_cfg.get("random_seed", 42))
    process = str(exp_cfg.get("process", "kfold")).strip().lower()
    x, y = data.x, data.y.astype(float)
    rows: list[dict[str, Any]] = []

    if process == "repeat":
        repeat_cfg = exp_cfg.get("repeat", {})
        split = float(repeat_cfg.get("split", 0.3))
        repeats = int(repeat_cfg.get("repeats", 15))
        for i in range(repeats):
            x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                test_size=split,
                random_state=random_seed + i,
                shuffle=True,
            )
            estimator = build_regressor_with_config(config, x_train, y_train)
            estimator.fit(x_train, y_train)
            y_pred = np.clip(estimator.predict(x_test), 0.0, 1.0)
            rows.append({"row_type": "split", "split_index": int(i), **evaluate_regressor(y_test, y_pred)})
        return pd.DataFrame(rows)

    if process == "kfold":
        kfold_cfg = exp_cfg.get("kfold", {})
        num_fold = int(kfold_cfg.get("num_fold", 10))
        kfold = KFold(n_splits=num_fold, shuffle=True, random_state=random_seed)
        for i, (train_idx, test_idx) in enumerate(kfold.split(x)):
            x_train, x_test = x[train_idx], x[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            estimator = build_regressor_with_config(config, x_train, y_train)
            estimator.fit(x_train, y_train)
            y_pred = np.clip(estimator.predict(x_test), 0.0, 1.0)
            rows.append({"row_type": "split", "split_index": int(i), **evaluate_regressor(y_test, y_pred)})
        return pd.DataFrame(rows)

    raise ValueError("experiment.process must be 'kfold' or 'repeat'.")


def suffix_name(column: str) -> str:
    if "__" not in column:
        return column
    return column.rsplit("__", 1)[-1]


def unto_o_feature_sets(features: list[str]) -> dict[str, list[str]]:
    by_suffix: dict[str, list[str]] = {}
    for col in features:
        by_suffix.setdefault(suffix_name(col), []).append(col)

    prob_items: list[tuple[int, str]] = []
    for suffix, cols in by_suffix.items():
        match = re.fullmatch(r"prob_(\d+)", suffix)
        if match:
            prob_items.extend((int(match.group(1)), col) for col in cols)
    prob_cols = [col for _idx, col in sorted(prob_items)]
    final_class = prob_cols + by_suffix.get("prob_sum", [])
    final_score = by_suffix.get("final_score", [])
    final_loc = [col for name in ["size", "circum", "size_circum"] for col in by_suffix.get(name, [])]
    target_class = by_suffix.get("cls_loss", [])
    target_score = by_suffix.get("obj_loss", [])
    target_loc = [
        col
        for name in ["x_loss", "y_loss", "w_loss", "h_loss", "size_diff", "circum_diff", "size_circum_diff"]
        for col in by_suffix.get(name, [])
    ]
    return {
        "Full UnTO-O": list(features),
        "Final-detection only": final_class + final_score + final_loc,
        "Target-relative only": target_class + target_score + target_loc,
        "Final class+score": final_class + final_score,
        "Final localization": final_loc,
        "Target class": target_class,
        "Target score": target_score,
        "Target localization": target_loc,
    }


def find_component_root(grid_root: Path, suffix: str) -> Path:
    matches = sorted(path for path in grid_root.iterdir() if path.is_dir() and path.name.endswith(suffix))
    if not matches:
        raise FileNotFoundError(f"No gradient component directory ending with {suffix!r} under {grid_root}")
    if len(matches) > 1:
        raise ValueError(
            f"Multiple gradient component directories ending with {suffix!r} under {grid_root}: "
            f"{[p.name for p in matches]}"
        )
    csv_path = matches[0] / "layer_grad.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    return matches[0]


def summarize_result(
    dataset: str,
    name: str,
    cls_data: LoadedTaskData,
    reg_data: LoadedTaskData,
    cls_eval: pd.DataFrame,
    reg_eval: pd.DataFrame,
) -> EvalResult:
    y = cls_data.y.astype(int)

    def _std(df: pd.DataFrame, col: str) -> float:
        return float(df[col].std(ddof=1)) if len(df) > 1 else 0.0

    return EvalResult(
        dataset=dataset,
        name=name,
        num_rows=int(len(cls_data.df)),
        num_features=int(cls_data.x.shape[1]),
        tp_ratio=float(np.mean(y)) if len(y) else 0.0,
        auroc_mean=float(cls_eval["auroc"].mean()),
        auroc_std=_std(cls_eval, "auroc"),
        ap_mean=float(cls_eval["ap"].mean()),
        ap_std=_std(cls_eval, "ap"),
        fpr95_mean=float(cls_eval["fpr95"].mean()),
        fpr95_std=_std(cls_eval, "fpr95"),
        r2_mean=float(reg_eval["r2"].mean()),
        r2_std=_std(reg_eval, "r2"),
        ece_mean=float(cls_eval["ece"].mean()) if "ece" in cls_eval else math.nan,
        ece_std=_std(cls_eval, "ece") if "ece" in cls_eval else math.nan,
        ace_mean=float(cls_eval["ace"].mean()) if "ace" in cls_eval else math.nan,
        ace_std=_std(cls_eval, "ace") if "ace" in cls_eval else math.nan,
    )


def evaluate_variant(
    dataset_name: str,
    variant_name: str,
    cls_data: LoadedTaskData,
    reg_data: LoadedTaskData,
    classifier_config: dict[str, Any],
    regressor_config: dict[str, Any],
) -> EvalResult:
    print(f"[{dataset_name}] {variant_name}: {cls_data.x.shape[1]} features, {len(cls_data.df)} rows")
    cls_eval = evaluate_classifier_protocol(cls_data, classifier_config)
    reg_eval = evaluate_regressor_protocol(reg_data, regressor_config)
    return summarize_result(dataset_name, variant_name, cls_data, reg_data, cls_eval, reg_eval)


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
        "ece_mean": result.ece_mean,
        "ece_std": result.ece_std,
        "ace_mean": result.ace_mean,
        "ace_std": result.ace_std,
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


def config_summary_rows(
    classifier_config_path: Path,
    regressor_config_path: Path,
    classifier_config: dict[str, Any],
    regressor_config: dict[str, Any],
) -> list[dict[str, str]]:
    clf_model = classifier_config.get("model", {})
    clf_exp = classifier_config.get("experiment", {})
    reg_model = regressor_config.get("model", {})
    reg_exp = regressor_config.get("experiment", {})
    clf_repeat = clf_exp.get("repeat", {})
    reg_repeat = reg_exp.get("repeat", {})
    return [
        {"item": "Classifier config", "setting": path_for_config(classifier_config_path)},
        {"item": "Classifier", "setting": f"{clf_model.get('type', 'gb_classifier')} ({clf_model.get('device', 'cpu')})"},
        {"item": "Classifier process", "setting": str(clf_exp.get("process", "kfold"))},
        {"item": "Classifier repeats", "setting": str(clf_repeat.get("repeats", "--"))},
        {"item": "Classifier test ratio", "setting": str(clf_repeat.get("split", "--"))},
        {"item": "Classifier augmentation", "setting": str(clf_exp.get("augmentation", "none"))},
        {"item": "Regressor config", "setting": path_for_config(regressor_config_path)},
        {"item": "Regressor", "setting": f"{reg_model.get('type', 'gb_regressor')} ({reg_model.get('device', 'cpu')})"},
        {"item": "Regressor process", "setting": str(reg_exp.get("process", "kfold"))},
        {"item": "Regressor repeats", "setting": str(reg_repeat.get("repeats", "--"))},
        {"item": "Regressor test ratio", "setting": str(reg_repeat.get("split", "--"))},
        {"item": "Random seed", "setting": f"classifier {clf_model.get('random_seed', 42)}, regressor {reg_model.get('random_seed', 42)}"},
        {"item": "Search", "setting": f"classifier {bool(clf_model.get('search', False))}, regressor {bool(reg_model.get('search', False))}"},
    ]


def write_hyperparameters(path: Path, rows: list[dict[str, str]]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Meta-model and split settings read from the main training YAML files.}",
        r"\label{tab:app_hyperparameters}",
        r"\begin{tabular}{L{0.28\linewidth}L{0.62\linewidth}}",
        r"\toprule",
        r"Item & Setting \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(f"{escape_tex(row['item'])} & {escape_tex(row['setting'])} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dataset_summary_row(config: DatasetConfig, loaded: LoadedTaskData, limit_rows: int | None) -> dict[str, float | int | str]:
    return {
        "detector": "YOLOv5",
        "dataset": config.summary_name,
        "display_name": config.display_name,
        "detections": int(len(loaded.df)),
        "tp_ratio": float(loaded.y.astype(int).mean()) if len(loaded.y) else 0.0,
        "regression_target": "gt_iou" if loaded.label_col == "gt_iou" else loaded.label_col,
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
    parser.add_argument("--classifier-config", default=str(DEFAULT_CLASSIFIER_CONFIG))
    parser.add_argument("--regressor-config", default=str(DEFAULT_REGRESSOR_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--limit-rows", type=int, default=None, help="Limit loaded rows for a quick dry run only.")
    parser.add_argument("--skip-unto-o", action="store_true")
    parser.add_argument("--skip-unto-g", action="store_true")
    return parser.parse_args()


def evaluate_unto_o(
    config: DatasetConfig,
    classifier_base: dict[str, Any],
    regressor_base: dict[str, Any],
    limit_rows: int | None,
) -> tuple[list[EvalResult], LoadedTaskData]:
    classifier_cfg = dataset_config_from_base(classifier_base, config.unto_o_root, config.gt_root)
    regressor_cfg = dataset_config_from_base(regressor_base, config.unto_o_root, config.gt_root)
    cls_full = prepare_loaded_data(classifier_cfg, "classifier", limit_rows)
    reg_full = prepare_loaded_data(regressor_cfg, "regressor", limit_rows)
    feature_sets = unto_o_feature_sets(cls_full.features)
    reg_feature_sets = unto_o_feature_sets(reg_full.features)
    results: list[EvalResult] = []
    for name, cls_cols in feature_sets.items():
        reg_cols = reg_feature_sets.get(name, [])
        if not cls_cols or not reg_cols:
            raise ValueError(f"Feature subset {name!r} is empty for {config.display_name}.")
        cls_data = make_subset_data(cls_full, cls_cols)
        reg_data = make_subset_data(reg_full, reg_cols)
        results.append(evaluate_variant(config.display_name, name, cls_data, reg_data, classifier_cfg, regressor_cfg))
    return results, cls_full


def evaluate_unto_g(
    config: DatasetConfig,
    classifier_base: dict[str, Any],
    regressor_base: dict[str, Any],
    limit_rows: int | None,
) -> list[EvalResult]:
    component_roots = {name: find_component_root(config.grad_grid_root, suffix) for name, suffix in GRAD_COMPONENTS.items()}
    roots_by_name: dict[str, list[Path]] = {name: [root] for name, root in component_roots.items()}
    roots_by_name["All components"] = [component_roots[name] for name in GRAD_COMPONENTS]

    order = ["Localization gradient only", "Classification gradient only", "Detection-score gradient only", "All components"]
    results: list[EvalResult] = []
    for name in order:
        roots = roots_by_name[name]
        classifier_cfg = dataset_config_from_base(classifier_base, roots, config.gt_root)
        regressor_cfg = dataset_config_from_base(regressor_base, roots, config.gt_root)
        cls_data = prepare_loaded_data(classifier_cfg, "classifier", limit_rows)
        reg_data = prepare_loaded_data(regressor_cfg, "regressor", limit_rows)
        results.append(evaluate_variant(config.display_name, name, cls_data, reg_data, classifier_cfg, regressor_cfg))
    return results


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configs = selected_configs(args)

    classifier_config_path = resolve_path(args.classifier_config)
    regressor_config_path = resolve_path(args.regressor_config)
    classifier_base = load_yaml(classifier_config_path)
    regressor_base = load_yaml(regressor_config_path)

    hyper_rows = config_summary_rows(classifier_config_path, regressor_config_path, classifier_base, regressor_base)
    pd.DataFrame(hyper_rows).to_csv(output_dir / "meta_model_hyperparameters.csv", index=False)
    write_hyperparameters(output_dir / "hyperparameters_table.tex", hyper_rows)
    write_gradient_layers_table(output_dir / "gradient_layers_table.tex")

    dataset_rows: list[dict[str, float | int | str]] = []
    unto_o_results: list[EvalResult] = []
    unto_g_results: list[EvalResult] = []
    metadata: dict[str, Any] = {
        "selected_datasets": [config.key for config in configs],
        "limit_rows": args.limit_rows,
        "classifier_config_path": str(classifier_config_path),
        "regressor_config_path": str(regressor_config_path),
        "classifier_config": classifier_base,
        "regressor_config": regressor_base,
        "datasets": {},
    }

    for config in configs:
        print(f"[Dataset] {config.display_name}")
        loaded_for_summary: LoadedTaskData | None = None
        if not args.skip_unto_o:
            results, loaded_for_summary = evaluate_unto_o(config, classifier_base, regressor_base, args.limit_rows)
            unto_o_results.extend(results)
        if loaded_for_summary is None:
            summary_cfg = dataset_config_from_base(classifier_base, config.unto_o_root, config.gt_root)
            loaded_for_summary = prepare_loaded_data(summary_cfg, "classifier", args.limit_rows)
        dataset_rows.append(dataset_summary_row(config, loaded_for_summary, args.limit_rows))
        if not args.skip_unto_g:
            unto_g_results.extend(evaluate_unto_g(config, classifier_base, regressor_base, args.limit_rows))
        metadata["datasets"][config.key] = {
            "display_name": config.display_name,
            "gt_root": str(config.gt_root),
            "unto_o_root": str(config.unto_o_root),
            "grad_grid_root": str(config.grad_grid_root),
            "loaded_rows": int(len(loaded_for_summary.df)),
            "unto_o_root_info": loaded_for_summary.root_info,
        }

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
