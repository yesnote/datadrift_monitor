from __future__ import annotations

import json
import pickle
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from tqdm import tqdm

from error_detectors.losses.loss import evaluate_classifier
from error_detectors.models.error_detector import build_estimator, param_grid

try:
    from imblearn.over_sampling import SMOTE
except Exception:  # pragma: no cover
    SMOTE = None

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class FeatureSpec:
    grad_columns: list[str]
    dim_by_column: dict[str, int]


def resolve_path_value(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def flatten_numeric(obj: Any) -> list[float]:
    if obj is None:
        return []
    if isinstance(obj, (int, float, np.integer, np.floating)):
        return [float(obj)]
    if isinstance(obj, str):
        text = obj.strip()
        if not text:
            return []
        try:
            return flatten_numeric(json.loads(text))
        except Exception:
            return []
    if isinstance(obj, dict):
        out: list[float] = []
        for key in sorted(obj.keys()):
            out.extend(flatten_numeric(obj[key]))
        return out
    if isinstance(obj, (list, tuple)):
        out: list[float] = []
        for item in obj:
            out.extend(flatten_numeric(item))
        return out
    return []


def infer_feature_spec(df: pd.DataFrame, grad_columns: list[str]) -> FeatureSpec:
    dim_by_column: dict[str, int] = {}
    for col in grad_columns:
        inferred_dim = 0
        for value in df[col].values:
            vec = flatten_numeric(value)
            if vec:
                inferred_dim = len(vec)
                break
        dim_by_column[col] = inferred_dim
    return FeatureSpec(grad_columns=grad_columns, dim_by_column=dim_by_column)


def build_feature_matrix(df: pd.DataFrame, spec: FeatureSpec) -> np.ndarray:
    rows: list[list[float]] = []
    for _, row in df.iterrows():
        feature_row: list[float] = []
        for col in spec.grad_columns:
            vec = flatten_numeric(row[col])
            dim = spec.dim_by_column[col]
            if dim == 0:
                continue
            if len(vec) < dim:
                vec = vec + [0.0] * (dim - len(vec))
            elif len(vec) > dim:
                vec = vec[:dim]
            feature_row.extend(vec)
        rows.append(feature_row)
    return np.asarray(rows, dtype=np.float32)


def apply_augmentation(x: np.ndarray, y: np.ndarray, augmentation: str) -> tuple[np.ndarray, np.ndarray]:
    if augmentation == "none":
        return x, y
    if augmentation == "smote":
        if SMOTE is None:
            raise ImportError("imblearn is required for augmentation='smote'.")
        sampler = SMOTE()
        return sampler.fit_resample(x, y)
    raise ValueError(f"Unsupported augmentation: {augmentation}")


def save_object(obj: Any, path_without_suffix: Path) -> Path:
    if joblib is not None:
        out = path_without_suffix.with_suffix(".joblib")
        joblib.dump(obj, out)
        return out
    out = path_without_suffix.with_suffix(".pkl")
    with open(out, "wb") as f:
        pickle.dump(obj, f)
    return out


def parse_root_info(root_path: Path) -> tuple[str, str]:
    model_group = root_path.parent.name
    run_name = root_path.name
    match = re.match(r"^\d{2}-\d{2}-\d{4}_\d{2};\d{2}_(.+)$", run_name)
    cue = match.group(1) if match else run_name.split("_", 2)[-1]
    return model_group, cue


def load_training_dataframe(dataset_cfg: dict[str, Any]) -> tuple[pd.DataFrame, str, list[str], dict[str, str]]:
    input_root_raw = str(dataset_cfg.get("input_root", "")).strip()
    gt_root_raw = str(dataset_cfg.get("gt_root", "")).strip()
    if not input_root_raw or not gt_root_raw:
        raise ValueError("dataset.input_root and dataset.gt_root are required.")

    input_root = resolve_path_value(input_root_raw)
    gt_root = resolve_path_value(gt_root_raw)
    input_group, input_cue = parse_root_info(input_root)
    gt_group, gt_cue = parse_root_info(gt_root)
    if input_group != gt_group:
        msg = (
            "dataset.input_root and dataset.gt_root must have the same model group "
            f"(got '{input_group}' vs '{gt_group}')."
        )
        warnings.warn(msg)
        raise ValueError(msg)

    input_csv_name = "layer_grad.csv" if input_cue == "layer_grad" else "feature_grad.csv"
    input_csv = input_root / input_csv_name
    if not input_csv.is_file():
        raise FileNotFoundError(f"{input_csv_name} not found: {input_csv}")
    feature_df = pd.read_csv(input_csv)

    if input_group == "fn_detectors":
        gt_csv = gt_root / "fn.csv"
        label_col = "fn"
        if not gt_csv.is_file():
            raise FileNotFoundError(f"fn.csv not found: {gt_csv}")
        gt_df = pd.read_csv(gt_csv)[["image_id", "image_path", label_col]]
        merge_keys = ["image_id", "image_path"]
        meta_columns = {"image_id", "image_path", label_col}
    elif input_group == "tp_classifiers":
        gt_csv = gt_root / "tp.csv"
        label_col = "tp"
        if not gt_csv.is_file():
            raise FileNotFoundError(f"tp.csv not found: {gt_csv}")
        gt_df = pd.read_csv(gt_csv)

        if {"image_id", "image_path", "pred_idx"}.issubset(feature_df.columns) and {
            "image_id",
            "image_path",
            "pred_idx",
            label_col,
        }.issubset(gt_df.columns):
            merge_keys = ["image_id", "image_path", "pred_idx"]
        elif {
            "image_id",
            "image_path",
            "xmin",
            "ymin",
            "xmax",
            "ymax",
        }.issubset(feature_df.columns) and {
            "image_id",
            "image_path",
            "xmin",
            "ymin",
            "xmax",
            "ymax",
            label_col,
        }.issubset(gt_df.columns):
            merge_keys = ["image_id", "image_path", "xmin", "ymin", "xmax", "ymax"]
        else:
            raise ValueError("Cannot match feature_grad.csv and tp.csv. Missing join keys.")

        keep_cols = list(dict.fromkeys(merge_keys + [label_col]))
        gt_df = gt_df[keep_cols]
        meta_columns = {
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
            "max_iou",
            label_col,
        }
    else:
        raise ValueError(
            "Unsupported model group from dataset roots: "
            f"'{input_group}'. Expected 'fn_detectors' or 'tp_classifiers'."
        )

    merged = feature_df.merge(gt_df, on=merge_keys, how="inner")
    if merged.empty:
        raise ValueError("Merged training dataframe is empty. Check input_root and gt_root pair.")
    if label_col not in merged.columns:
        raise ValueError(f"Ground-truth label column '{label_col}' is missing after merge.")

    grad_columns = [c for c in merged.columns if c not in meta_columns]
    if not grad_columns:
        raise ValueError("No gradient feature columns found after merge.")

    root_info = {
        "input_root": str(input_root),
        "gt_root": str(gt_root),
        "model_group": input_group,
        "input_cue": input_cue,
        "gt_cue": gt_cue,
    }
    return merged, label_col, grad_columns, root_info


def run_train(config: dict[str, Any], run_dir: Path) -> Path:
    mode = str(config.get("mode", "train"))
    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    train_cfg = config["train"]

    out_dir = run_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df, label_col, grad_columns, root_info = load_training_dataframe(dataset_cfg)
    y = df[label_col].astype(int).to_numpy()
    spec = infer_feature_spec(df, grad_columns)
    x = build_feature_matrix(df, spec)

    model_name = str(model_cfg.get("type", "gb_classifier"))
    device = str(model_cfg.get("device", "cpu"))
    estimator = build_estimator(model_name, device=device)
    best_params: dict[str, Any] = {}

    do_search = bool(model_cfg.get("search", False))
    augmentation = str(train_cfg.get("augmentation", "none"))
    if do_search:
        x_search, y_search = apply_augmentation(x, y, augmentation)
        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid(model_name),
            scoring=str(model_cfg.get("search_scoring", "roc_auc")),
            n_jobs=int(train_cfg.get("n_jobs", 8)),
            cv=5,
            verbose=1,
        )
        search.fit(x_search, y_search)
        best_params = dict(search.best_params_)
        estimator.set_params(**best_params)

    repeats = int(train_cfg.get("repeats", 15))
    test_size = float(train_cfg.get("test_size", 0.3))
    random_seed = int(train_cfg.get("random_seed", 42))

    eval_rows: list[dict[str, float]] = []
    split_iter = tqdm(range(repeats), desc=f"Error Detector ({mode})", unit="split")
    for i in split_iter:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=random_seed + i,
            stratify=y,
        )
        x_train, y_train = apply_augmentation(x_train, y_train, augmentation)

        estimator.fit(x_train, y_train)
        y_pred = estimator.predict_proba(x_test)[:, 1]

        acc, auroc = evaluate_classifier(y_test, y_pred)
        eval_rows.append({"accuracy": float(acc), "auroc": float(auroc)})

        pd.DataFrame({"y_test": y_test, "y_pred": y_pred}).to_csv(out_dir / f"eval_data_{i}.csv", index=False)
        save_object(estimator, out_dir / f"model_{i}")

    eval_df = pd.DataFrame(eval_rows)
    eval_df.loc["mean"] = eval_df.mean(numeric_only=True)
    eval_df.loc["std"] = eval_df.std(numeric_only=True)
    eval_df.to_csv(out_dir / "evaluation_results.csv", index=True)

    metadata = {
        "input_root": root_info["input_root"],
        "gt_root": root_info["gt_root"],
        "model_group": root_info["model_group"],
        "input_cue": root_info["input_cue"],
        "gt_cue": root_info["gt_cue"],
        "label_column": label_col,
        "model": model_name,
        "device": device,
        "search_scoring": str(model_cfg.get("search_scoring", "roc_auc")),
        "augmentation": augmentation,
        "feature_dimension": int(x.shape[1]),
        "num_rows": int(len(df)),
        "num_positive_fn": int(np.sum(y)),
        "grad_columns": grad_columns,
        "dim_by_column": spec.dim_by_column,
        "best_params": best_params,
        "repeats": repeats,
        "test_size": test_size,
        "random_seed": random_seed,
        "search": do_search,
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Saved outputs to: {out_dir}")
    print(eval_df)
    return out_dir
