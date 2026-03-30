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
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
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


def sanitize_feature_matrix(x: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    stats = {
        "nan_count": int(np.isnan(x).sum()),
        "posinf_count": int(np.isposinf(x).sum()),
        "neginf_count": int(np.isneginf(x).sum()),
    }
    if stats["nan_count"] == 0 and stats["posinf_count"] == 0 and stats["neginf_count"] == 0:
        return x, stats
    # Replace non-finite values with bounded finite numbers to keep sklearn pipeline stable.
    x_clean = np.nan_to_num(
        x,
        nan=0.0,
        posinf=1e6,
        neginf=-1e6,
    ).astype(np.float32, copy=False)
    return x_clean, stats


def apply_augmentation(
    x: np.ndarray,
    y: np.ndarray,
    augmentation: str,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    if augmentation == "none":
        return x, y
    if augmentation == "smote":
        if SMOTE is None:
            raise ImportError("imblearn is required for augmentation='smote'.")
        sampler = SMOTE(random_state=int(random_seed))
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


def parse_root_info(root_path: Path) -> tuple[str, str, str]:
    # Current format: .../runs/{model_group}/{time}_{cue}_{target?}
    # Legacy format:  .../runs/{model_group}/{cue}/{time}
    # Legacy format:  .../runs/{model_group}/{time}_{cue}
    parent = root_path.parent
    if parent.name in {"fn_detectors", "tp_classifiers"}:
        model_group = parent.name
        run_name = root_path.name
        match = re.match(r"^\d{2}-\d{2}-\d{4}_\d{2};\d{2}_(.+)$", run_name)
        tail = match.group(1) if match else run_name
        for cue_name in ("feature_grad", "layer_grad", "full_softmax", "mc_dropout", "meta_detect", "entropy", "energy", "feature", "score", "gt", "fn", "tp"):
            if tail == cue_name:
                return model_group, cue_name, ""
            prefix = f"{cue_name}_"
            if tail.startswith(prefix):
                return model_group, cue_name, tail[len(prefix):]
        return model_group, tail, ""

    if parent.parent.name in {"fn_detectors", "tp_classifiers"}:
        model_group = parent.parent.name
        cue = parent.name
        return model_group, cue, ""

    raise ValueError(
        "dataset root must follow object_detectors/runs/{fn_detectors|tp_classifiers}/{time}_{cue}_{target?} "
        "or legacy formats."
    )


def normalize_input_roots(raw_value: Any) -> list[str]:
    if isinstance(raw_value, str):
        value = raw_value.strip()
        return [value] if value else []
    if isinstance(raw_value, (list, tuple)):
        out: list[str] = []
        for item in raw_value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    return []


def load_training_dataframe(dataset_cfg: dict[str, Any]) -> tuple[pd.DataFrame, str, list[str], dict[str, Any]]:
    input_root_raw_list = normalize_input_roots(dataset_cfg.get("input_root", ""))
    gt_root_raw = str(dataset_cfg.get("gt_root", "")).strip()
    if not input_root_raw_list or not gt_root_raw:
        raise ValueError("dataset.input_root (str or list[str]) and dataset.gt_root are required.")

    input_roots = [resolve_path_value(v) for v in input_root_raw_list]
    gt_root = resolve_path_value(gt_root_raw)
    input_infos = [parse_root_info(p) for p in input_roots]
    input_groups = {group for group, _cue, _target in input_infos}
    if len(input_groups) != 1:
        msg = f"All dataset.input_root entries must share one model group, got: {sorted(input_groups)}"
        warnings.warn(msg)
        raise ValueError(msg)
    input_group = next(iter(input_groups))

    gt_group, _gt_cue, _gt_target = parse_root_info(gt_root)
    if input_group != gt_group:
        msg = (
            "dataset.input_root and dataset.gt_root must have the same model group "
            f"(got '{input_group}' vs '{gt_group}')."
        )
        warnings.warn(msg)
        raise ValueError(msg)

    cue_to_csv = {
        "feature": "feature.csv",
        "layer_grad": "layer_grad.csv",
        "feature_grad": "feature_grad.csv",
        "score": "score.csv",
        "mc_dropout": "mc_dropout.csv",
        "meta_detect": "meta_detect.csv",
        "full_softmax": "full_softmax.csv",
        "entropy": "entropy.csv",
        "energy": "energy.csv",
    }
    if input_group == "fn_detectors":
        gt_csv = gt_root / "fn.csv"
        label_col = "fn"
        if not gt_csv.is_file():
            raise FileNotFoundError(f"fn.csv not found: {gt_csv}")
        gt_df = pd.read_csv(gt_csv)[["image_id", "image_path", label_col]]
        base_merge_keys = ["image_id", "image_path"]
    elif input_group == "tp_classifiers":
        gt_csv = gt_root / "tp.csv"
        label_col = "tp"
        if not gt_csv.is_file():
            raise FileNotFoundError(f"tp.csv not found: {gt_csv}")
        gt_df = pd.read_csv(gt_csv)
        if {"image_id", "image_path", "pred_idx", label_col}.issubset(gt_df.columns):
            base_merge_keys = ["image_id", "image_path", "pred_idx"]
        elif {"image_id", "image_path", "xmin", "ymin", "xmax", "ymax", label_col}.issubset(gt_df.columns):
            base_merge_keys = ["image_id", "image_path", "xmin", "ymin", "xmax", "ymax"]
        else:
            raise ValueError("tp.csv missing required join keys.")
        gt_df = gt_df[list(dict.fromkeys(base_merge_keys + [label_col]))]
    else:
        raise ValueError(
            "Unsupported model group from dataset roots: "
            f"'{input_group}'. Expected 'fn_detectors' or 'tp_classifiers'."
        )

    merged = gt_df.copy()
    prefixed_feature_columns: list[str] = []
    input_uncertainties: list[str] = []
    input_targets: list[str] = []
    for input_root, (_group, input_cue, input_target) in zip(input_roots, input_infos):
        input_csv_name = cue_to_csv.get(input_cue)
        if input_csv_name is None:
            raise ValueError(
                f"Unsupported input uncertainty '{input_cue}'. "
                "Supported uncertainties: feature, layer_grad, feature_grad, score, mc_dropout, meta_detect, full_softmax, entropy, energy."
            )
        input_csv = input_root / input_csv_name
        if not input_csv.is_file():
            raise FileNotFoundError(f"{input_csv_name} not found: {input_csv}")
        feature_df = pd.read_csv(input_csv)

        if input_group == "fn_detectors":
            merge_keys = ["image_id", "image_path"]
            meta_columns = {"image_id", "image_path", "num_preds", label_col}
        else:
            if {"image_id", "image_path", "pred_idx"}.issubset(feature_df.columns) and set(base_merge_keys) == {
                "image_id",
                "image_path",
                "pred_idx",
            }:
                merge_keys = ["image_id", "image_path", "pred_idx"]
            elif {
                "image_id",
                "image_path",
                "xmin",
                "ymin",
                "xmax",
                "ymax",
            }.issubset(feature_df.columns) and set(base_merge_keys) == {
                "image_id",
                "image_path",
                "xmin",
                "ymin",
                "xmax",
                "ymax",
            }:
                merge_keys = ["image_id", "image_path", "xmin", "ymin", "xmax", "ymax"]
            else:
                raise ValueError(
                    f"Cannot match {input_csv_name} to tp.csv using keys {base_merge_keys}. "
                    "All inputs must use the same tp key type."
                )
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
            if input_cue == "score":
                meta_columns.discard("score")

        feature_columns = [c for c in feature_df.columns if c not in meta_columns]
        if not feature_columns:
            raise ValueError(f"No input feature columns found in {input_csv}")
        suffix_target = f"_{input_target}" if input_target else ""
        prefix = f"{input_cue}{suffix_target}__"
        rename_map = {c: f"{prefix}{c}" for c in feature_columns}
        feature_df = feature_df[list(dict.fromkeys(merge_keys + feature_columns))].rename(columns=rename_map)
        merged = merged.merge(feature_df, on=merge_keys, how="inner")
        prefixed_feature_columns.extend(rename_map.values())
        input_uncertainties.append(input_cue)
        input_targets.append(input_target)

    if merged.empty:
        raise ValueError("Merged training dataframe is empty. Check input_root and gt_root pair.")
    if label_col not in merged.columns:
        raise ValueError(f"Ground-truth label column '{label_col}' is missing after merge.")

    input_features = [c for c in merged.columns if c in prefixed_feature_columns]
    if not input_features:
        raise ValueError("No input feature columns found after merge.")

    root_info = {
        "input_root": [str(p) for p in input_roots],
        "gt_root": str(gt_root),
        "model_group": input_group,
        "input_uncertainty": input_uncertainties,
        "input_target": input_targets,
    }
    return merged, label_col, input_features, root_info


def run_train(config: dict[str, Any], run_dir: Path) -> Path:
    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    exp_cfg = config["experiment"]

    out_dir = run_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df, label_col, grad_columns, root_info = load_training_dataframe(dataset_cfg)
    y = df[label_col].astype(int).to_numpy()
    spec = infer_feature_spec(df, grad_columns)
    x = build_feature_matrix(df, spec)
    x, nonfinite_stats = sanitize_feature_matrix(x)
    had_nonfinite = any(v > 0 for v in nonfinite_stats.values())
    if had_nonfinite:
        warnings.warn(
            "Non-finite values found in input features and replaced with finite values: "
            f"{nonfinite_stats}"
        )

    model_name = str(model_cfg.get("type", "gb_classifier"))
    device = str(model_cfg.get("device", "cpu"))
    random_seed = int(model_cfg.get("random_seed", 42))
    estimator = build_estimator(model_name, device=device, random_seed=random_seed)
    best_params: dict[str, Any] = {}

    do_search = bool(model_cfg.get("search", False))
    augmentation = str(exp_cfg.get("augmentation", "none"))
    if do_search:
        x_search, y_search = apply_augmentation(x, y, augmentation, random_seed=random_seed)
        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid(model_name),
            scoring=str(model_cfg.get("search_scoring", "roc_auc")),
            n_jobs=int(exp_cfg.get("n_jobs", 8)),
            cv=5,
            verbose=1,
        )
        search.fit(x_search, y_search)
        best_params = dict(search.best_params_)
        estimator.set_params(**best_params)

    eval_rows: list[dict[str, float]] = []
    process = str(exp_cfg.get("process", "kfold")).strip().lower()
    used_num_fold = None
    used_split = None
    used_repeats = None

    if process == "kfold":
        kfold_cfg = exp_cfg.get("kfold", {})
        num_fold = int(kfold_cfg.get("num_fold", 10))
        if num_fold < 2:
            raise ValueError("experiment.kfold.num_fold must be >= 2.")
        used_num_fold = num_fold
        kfold = StratifiedKFold(n_splits=num_fold, shuffle=True, random_state=random_seed)
        split_iter = tqdm(enumerate(kfold.split(x, y)), desc="Error Detector (kfold)", total=num_fold, unit="fold")
        for i, (train_idx, test_idx) in split_iter:
            x_train, x_test = x[train_idx], x[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            x_train, y_train = apply_augmentation(x_train, y_train, augmentation, random_seed=random_seed + i)

            estimator.fit(x_train, y_train)
            y_pred = estimator.predict_proba(x_test)[:, 1]

            auroc, ap = evaluate_classifier(y_test, y_pred)
            eval_rows.append({"auroc": float(auroc), "ap": float(ap)})

            pd.DataFrame({"y_test": y_test, "y_pred": y_pred}).to_csv(out_dir / f"eval_data_{i}.csv", index=False)
            save_object(estimator, out_dir / f"model_{i}")
    elif process == "repeat":
        repeat_cfg = exp_cfg.get("repeat", {})
        split = float(repeat_cfg.get("split", 0.3))
        repeats = int(repeat_cfg.get("repeats", 15))
        if not (0.0 < split < 1.0):
            raise ValueError("experiment.repeat.split must be in (0,1).")
        if repeats < 1:
            raise ValueError("experiment.repeat.repeats must be >= 1.")
        used_split = split
        used_repeats = repeats
        split_iter = tqdm(range(repeats), desc="Error Detector (repeat)", total=repeats, unit="split")
        for i in split_iter:
            x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                test_size=split,
                random_state=random_seed + i,
                stratify=y,
                shuffle=True,
            )
            x_train, y_train = apply_augmentation(x_train, y_train, augmentation, random_seed=random_seed + i)

            estimator.fit(x_train, y_train)
            y_pred = estimator.predict_proba(x_test)[:, 1]

            auroc, ap = evaluate_classifier(y_test, y_pred)
            eval_rows.append({"auroc": float(auroc), "ap": float(ap)})

            pd.DataFrame({"y_test": y_test, "y_pred": y_pred}).to_csv(out_dir / f"eval_data_{i}.csv", index=False)
            save_object(estimator, out_dir / f"model_{i}")
    else:
        raise ValueError("experiment.process must be 'kfold' or 'repeat'.")

    eval_df = pd.DataFrame(eval_rows)
    eval_df.loc["mean"] = eval_df.mean(numeric_only=True)
    eval_df.loc["std"] = eval_df.std(numeric_only=True)
    eval_df.to_csv(out_dir / "evaluation_results.csv", index=True)

    metadata = {
        "input_root": root_info["input_root"],
        "gt_root": root_info["gt_root"],
        "model_group": root_info["model_group"],
        "input_uncertainty": root_info["input_uncertainty"],
        "input_target": root_info.get("input_target", ""),
        "label_column": label_col,
        "model": model_name,
        "device": device,
        "search_scoring": str(model_cfg.get("search_scoring", "roc_auc")),
        "augmentation": augmentation,
        "feature_dimension": int(x.shape[1]),
        "num_rows": int(len(df)),
        "num_positive_fn": int(np.sum(y)),
        "input_features": grad_columns,
        "dim_by_feature": spec.dim_by_column,
        "best_params": best_params,
        "process": process,
        "num_fold": used_num_fold,
        "repeat_split": used_split,
        "repeat_repeats": used_repeats,
        "nonfinite_replaced": had_nonfinite,
        "nonfinite_stats": nonfinite_stats,
        "random_seed": random_seed,
        "shuffle": True,
        "search": do_search,
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Saved outputs to: {out_dir}")
    print(eval_df)
    return out_dir
