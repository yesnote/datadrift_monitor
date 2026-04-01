from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from error_detectors.losses.loss import evaluate_classifier
from error_detectors.commands.run_train import (
    FeatureSpec,
    build_feature_matrix,
    infer_feature_spec,
    load_object,
    load_training_dataframe,
    sanitize_feature_matrix,
)


def _parse_model_index(path: Path) -> int:
    match = re.match(r"^model_(\d+)\.(joblib|pkl)$", path.name)
    if not match:
        return 10**9
    return int(match.group(1))


def _resolve_model_paths(model_cfg: dict[str, Any]) -> list[Path]:
    model_path_raw = str(model_cfg.get("model_path", "")).strip()
    if model_path_raw:
        model_path = Path(model_path_raw).resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"model.model_path not found: {model_path}")
        return [model_path]

    model_root_raw = str(model_cfg.get("model_root", "")).strip()
    if not model_root_raw:
        raise ValueError("For mode='test', set model.model_path or model.model_root in config.")
    model_root = Path(model_root_raw).resolve()
    if not model_root.is_dir():
        raise FileNotFoundError(f"model.model_root not found: {model_root}")

    model_index_raw = model_cfg.get("model_index", None)
    if model_index_raw is not None:
        model_index = int(model_index_raw)
        candidates = [
            model_root / f"model_{model_index}.joblib",
            model_root / f"model_{model_index}.pkl",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return [candidate]
        raise FileNotFoundError(
            f"model_index={model_index} requested but model_{model_index}.joblib/.pkl not found in {model_root}"
        )

    paths = [*model_root.glob("model_*.joblib"), *model_root.glob("model_*.pkl")]
    paths = sorted({p.resolve() for p in paths}, key=_parse_model_index)
    if not paths:
        raise FileNotFoundError(f"No model_*.joblib/.pkl files found in {model_root}")
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

    eval_df = pd.DataFrame(eval_rows)
    summary = {
        "model_file": "mean/std",
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
