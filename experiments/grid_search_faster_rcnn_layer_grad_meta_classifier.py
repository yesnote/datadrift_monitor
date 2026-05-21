import csv
import itertools
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

# Edit these paths before running.
OBJECT_DETECTOR_CONFIG = r"object_detectors/configs/faster_rcnn/predict_coco_faster_rcnn.yaml"
META_CLASSIFIER_CONFIG = r"meta_models/meta_classifier/configs/train_meta_classifier.yaml"

# If empty, dataset.gt_root is read from META_CLASSIFIER_CONFIG.
GT_ROOT = ""

# Set to a fixed name to resume a previous grid root. Empty string creates a new timestamped root.
GRID_NAME = ""

RUN_LAYER_GRAD = True
RUN_META_CLASSIFIER = True
REUSE_EXISTING = False

# Use None for all valid combinations, or set a small int for a smoke test.
MAX_COMBINATIONS = None

SCALARS = ["rpn_obj_loss", "rpn_bbox_loss", "roi_cls_loss", "roi_bbox_loss"]
TARGETS = ["cand_target", "null_target"]

RPN_BBOX_LOSSES = ["l1", "l2"]
RPN_BBOX_DIRECTIONS = ["pred_to_target"]
RPN_OBJ_LOSSES = ["bcewithlogits", "abs_diff", "signed_diff"]

ROI_BBOX_LOSSES = ["l1", "l2"]
ROI_BBOX_DIRECTIONS = ["pred_to_target"]
ROI_CLS_LOSSES = ["bcewithlogits", "kl", "ce"]


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _save_yaml(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def _resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def _dataset_name(config: dict) -> str:
    raw = config.get("dataset", {}).get("used_dataset", "unknown")
    if isinstance(raw, str):
        names = [raw.strip().lower()]
    elif isinstance(raw, (list, tuple)):
        names = [str(v).strip().lower() for v in raw if str(v).strip()]
    else:
        names = []
    if not names:
        return "unknown"
    return "-".join(
        "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in name)
        for name in names
    )


def _model_name(config: dict) -> str:
    raw = str(config.get("model", {}).get("type", "faster_rcnn")).strip().lower()
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in raw) or "faster_rcnn"


def _abbr(value: str) -> str:
    aliases = {
        "cand_target": "cand",
        "null_target": "null",
        "pred_to_target": "pred",
        "target_to_pred": "rev",
        "bcewithlogits": "bce",
        "abs_diff": "abs",
        "signed_diff": "signed",
    }
    return aliases.get(value, value)


def _valid_cls_directions(cls_loss: str) -> list[str]:
    return ["pred_to_target", "target_to_pred"] if cls_loss == "kl" else ["pred_to_target"]


def _valid_obj_directions(obj_loss: str) -> list[str]:
    if obj_loss == "signed_diff":
        return ["pred_to_target", "target_to_pred"]
    return ["pred_to_target"]


def _common_key(combo: dict) -> tuple:
    return (
        combo["rpn_bbox_loss"],
        combo["rpn_bbox_direction"],
        combo["rpn_obj_loss"],
        combo["rpn_obj_direction"],
        combo["roi_bbox_loss"],
        combo["roi_bbox_direction"],
        combo["roi_cls_loss"],
        combo["roi_cls_direction"],
    )


def _pair_key(combo: dict) -> tuple:
    return _common_key(combo)


def _combo_slug(combo: dict, target: str) -> str:
    return (
        f"layer_grad_t-{_abbr(target)}"
        f"__rb-{combo['rpn_bbox_loss']}-{_abbr(combo['rpn_bbox_direction'])}"
        f"__roib-{combo['roi_bbox_loss']}-{_abbr(combo['roi_bbox_direction'])}"
        f"__roic-{_abbr(combo['roi_cls_loss'])}-{_abbr(combo['roi_cls_direction'])}"
        f"__rpno-{_abbr(combo['rpn_obj_loss'])}-{_abbr(combo['rpn_obj_direction'])}"
    )


def _timestamped_combo_dir(root: Path, slug: str) -> Path:
    existing = sorted(root.glob(f"??-??-????_??;??_{slug}"))
    if existing and REUSE_EXISTING:
        return existing[-1]
    return root / f"{datetime.now().strftime('%m-%d-%Y_%H;%M')}_{slug}"


def iter_combinations():
    count = 0
    for rpn_bbox_loss, rpn_obj_loss, roi_bbox_loss, roi_cls_loss in itertools.product(
        RPN_BBOX_LOSSES,
        RPN_OBJ_LOSSES,
        ROI_BBOX_LOSSES,
        ROI_CLS_LOSSES,
    ):
        for rpn_bbox_direction in RPN_BBOX_DIRECTIONS:
            for rpn_obj_direction in _valid_obj_directions(rpn_obj_loss):
                for roi_bbox_direction in ROI_BBOX_DIRECTIONS:
                    for roi_cls_direction in _valid_cls_directions(roi_cls_loss):
                        yield {
                            "rpn_bbox_loss": rpn_bbox_loss,
                            "rpn_bbox_direction": rpn_bbox_direction,
                            "rpn_obj_loss": rpn_obj_loss,
                            "rpn_obj_direction": rpn_obj_direction,
                            "roi_bbox_loss": roi_bbox_loss,
                            "roi_bbox_direction": roi_bbox_direction,
                            "roi_cls_loss": roi_cls_loss,
                            "roi_cls_direction": roi_cls_direction,
                        }
                        count += 1
                        if MAX_COMBINATIONS is not None and count >= int(MAX_COMBINATIONS):
                            return


def _run(cmd: list[str]) -> None:
    print(" ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _prepare_layer_grad_config(base_config: dict, combo: dict, target: str) -> dict:
    config = deepcopy(base_config)
    config["mode"] = "predict"
    output_cfg = config.setdefault("output", {})
    output_cfg["uncertainty"] = "layer_grad"

    layer_grad_cfg = output_cfg.setdefault("layer_grad", {})
    layer_grad_cfg.setdefault("save_csv", {})["enabled"] = True
    grad_cfg = layer_grad_cfg.setdefault("gradient", {})

    grad_cfg["target"] = target
    grad_cfg["scalar"] = list(SCALARS)
    grad_cfg["layer"] = {
        "rpn_obj_loss": ["rpn.head.conv", "rpn.head.cls_logits"],
        "rpn_bbox_loss": ["rpn.head.conv", "rpn.head.bbox_pred"],
        "roi_cls_loss": ["roi_heads.box_head.fc7", "roi_heads.box_predictor.cls_score"],
        "roi_bbox_loss": ["roi_heads.box_head.fc7", "roi_heads.box_predictor.bbox_pred"],
    }
    grad_cfg["rpn"] = {
        "cand_obj_threshold": 0.0,
        "obj_loss": combo["rpn_obj_loss"],
        "bbox_loss": combo["rpn_bbox_loss"],
        "obj_direction": combo["rpn_obj_direction"],
        "bbox_direction": combo["rpn_bbox_direction"],
    }
    grad_cfg["roi"] = {
        "cand_score_threshold": 0.0,
        "cls_loss": combo["roi_cls_loss"],
        "bbox_loss": combo["roi_bbox_loss"],
        "cls_direction": combo["roi_cls_direction"],
        "bbox_direction": combo["roi_bbox_direction"],
    }
    return config


def _prepare_meta_config(base_config: dict, layer_grad_run_dir: Path, gt_root: str) -> dict:
    config = deepcopy(base_config)
    config["mode"] = "train"
    dataset_cfg = config.setdefault("dataset", {})
    dataset_cfg["input_root"] = [str(layer_grad_run_dir)]
    if gt_root:
        dataset_cfg["gt_root"] = str(_resolve_path(gt_root))
    return config


def _read_mean_metrics(eval_csv: Path) -> dict:
    if not eval_csv.is_file():
        return {}
    with open(eval_csv, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    mean_row = None
    for row in rows:
        first_value = next(iter(row.values()), "")
        if str(first_value).strip().lower() == "mean":
            mean_row = row
            break
    if mean_row is None and rows:
        mean_row = rows[-1]
    out = {}
    for key in ("auroc", "ap", "ece", "ace"):
        try:
            out[key] = float(mean_row.get(key, "nan"))
        except Exception:
            out[key] = float("nan")
    return out


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run_layer_grad_if_needed(
    od_base_config: dict,
    od_grid_root: Path,
    combo: dict,
    target: str,
) -> Path:
    slug = _combo_slug(combo, target)
    layer_dir = _timestamped_combo_dir(od_grid_root, slug)
    layer_csv = layer_dir / "layer_grad.csv"
    if RUN_LAYER_GRAD and not (REUSE_EXISTING and layer_csv.is_file()):
        od_config = _prepare_layer_grad_config(od_base_config, combo, target)
        od_config_path = layer_dir / "grid_object_detector_config.yaml"
        _save_yaml(od_config, od_config_path)
        _run(
            [
                sys.executable,
                "object_detectors/main.py",
                "--config",
                str(od_config_path),
                "--run-dir",
                str(layer_dir),
            ]
        )
    return layer_dir


def _run_meta_if_needed(
    meta_base_config: dict,
    meta_grid_root: Path,
    combo: dict,
    target: str,
    layer_dir: Path,
    gt_root: str,
) -> Path:
    slug = _combo_slug(combo, target)
    meta_dir = _timestamped_combo_dir(meta_grid_root, slug)
    eval_csv = meta_dir / "results" / "evaluation_results.csv"
    if RUN_META_CLASSIFIER and not (REUSE_EXISTING and eval_csv.is_file()):
        meta_config = _prepare_meta_config(meta_base_config, layer_dir, gt_root)
        meta_config_path = meta_dir / "grid_meta_classifier_config.yaml"
        _save_yaml(meta_config, meta_config_path)
        _run(
            [
                sys.executable,
                "meta_models/meta_classifier/main.py",
                "--config",
                str(meta_config_path),
                "--run-dir",
                str(meta_dir),
            ]
        )
    return meta_dir


def main() -> None:
    od_config_path = _resolve_path(OBJECT_DETECTOR_CONFIG)
    meta_config_path = _resolve_path(META_CLASSIFIER_CONFIG)
    od_base_config = _load_yaml(od_config_path)
    meta_base_config = _load_yaml(meta_config_path)
    dataset = _dataset_name(od_base_config)
    model = _model_name(od_base_config)

    grid_name = GRID_NAME.strip() or f"{datetime.now().strftime('%m-%d-%Y_%H;%M')}_faster_rcnn_layer_grad_grid"
    od_grid_root = REPO_ROOT / "object_detectors" / "runs" / model / "predict" / dataset / grid_name
    meta_grid_root = REPO_ROOT / "meta_models" / "meta_classifier" / "runs" / "train" / model / dataset / grid_name
    od_grid_root.mkdir(parents=True, exist_ok=True)
    meta_grid_root.mkdir(parents=True, exist_ok=True)

    gt_root = GT_ROOT.strip() or str(meta_base_config.get("dataset", {}).get("gt_root", "")).strip()
    if not gt_root:
        raise ValueError("GT_ROOT is empty and META_CLASSIFIER_CONFIG has no dataset.gt_root.")

    rows = []
    pair_rows = []
    run_cache = {}
    result_seen = set()
    combo_list = list(iter_combinations())
    for idx, combo in enumerate(combo_list, start=1):
        print(f"[{idx}/{len(combo_list)}] {_combo_slug(combo, 'cand_target')} vs {_combo_slug(combo, 'null_target')}", flush=True)

        metrics_by_target = {}
        dirs_by_target = {}
        for target in TARGETS:
            cache_key = (target, _combo_slug(combo, target))
            if cache_key not in run_cache:
                layer_dir = _run_layer_grad_if_needed(od_base_config, od_grid_root, combo, target)
                meta_dir = _run_meta_if_needed(meta_base_config, meta_grid_root, combo, target, layer_dir, gt_root)
                run_cache[cache_key] = (layer_dir, meta_dir)
            layer_dir, meta_dir = run_cache[cache_key]
            eval_csv = meta_dir / "results" / "evaluation_results.csv"
            metrics = _read_mean_metrics(eval_csv)
            metrics_by_target[target] = metrics
            dirs_by_target[target] = (layer_dir, meta_dir)
            result_key = (target, _combo_slug(combo, target))
            if result_key not in result_seen:
                result_seen.add(result_key)
                rows.append(
                    {
                        "target": target,
                    "rpn_bbox_loss": combo["rpn_bbox_loss"],
                    "rpn_bbox_direction": combo["rpn_bbox_direction"],
                    "rpn_obj_loss": combo["rpn_obj_loss"],
                    "rpn_obj_direction": combo["rpn_obj_direction"],
                        "roi_bbox_loss": combo["roi_bbox_loss"],
                        "roi_bbox_direction": combo["roi_bbox_direction"],
                        "roi_cls_loss": combo["roi_cls_loss"],
                        "roi_cls_direction": combo["roi_cls_direction"],
                        "layer_grad_run_dir": str(layer_dir),
                        "meta_classifier_run_dir": str(meta_dir),
                        "auroc": metrics.get("auroc", ""),
                        "ap": metrics.get("ap", ""),
                        "ece": metrics.get("ece", ""),
                        "ace": metrics.get("ace", ""),
                    }
                )

        cand = metrics_by_target.get("cand_target", {})
        null = metrics_by_target.get("null_target", {})
        try:
            cand_auroc = float(cand.get("auroc", "nan"))
            null_auroc = float(null.get("auroc", "nan"))
            cand_ap = float(cand.get("ap", "nan"))
            null_ap = float(null.get("ap", "nan"))
        except Exception:
            cand_auroc = null_auroc = cand_ap = null_ap = float("nan")
        pair_rows.append(
            {
                "rpn_bbox_loss": combo["rpn_bbox_loss"],
                "rpn_bbox_direction": combo["rpn_bbox_direction"],
                "rpn_obj_loss": combo["rpn_obj_loss"],
                "rpn_obj_direction": combo["rpn_obj_direction"],
                "roi_bbox_loss": combo["roi_bbox_loss"],
                "roi_bbox_direction": combo["roi_bbox_direction"],
                "roi_cls_loss": combo["roi_cls_loss"],
                "roi_cls_direction": combo["roi_cls_direction"],
                "cand_auroc": cand_auroc,
                "null_auroc": null_auroc,
                "delta_auroc": null_auroc - cand_auroc,
                "cand_ap": cand_ap,
                "null_ap": null_ap,
                "delta_ap": null_ap - cand_ap,
                "null_better_auroc": int(null_auroc > cand_auroc),
                "null_better_ap": int(null_ap > cand_ap),
                "cand_layer_grad_run_dir": str(dirs_by_target["cand_target"][0]),
                "null_layer_grad_run_dir": str(dirs_by_target["null_target"][0]),
                "cand_meta_classifier_run_dir": str(dirs_by_target["cand_target"][1]),
                "null_meta_classifier_run_dir": str(dirs_by_target["null_target"][1]),
            }
        )
        _write_results(meta_grid_root, rows, pair_rows)

    print(f"Saved grid results: {meta_grid_root / 'grid_results.csv'}")
    print(f"Saved pair comparison: {meta_grid_root / 'target_pair_comparison.csv'}")
    print(f"Saved null comparison: {meta_grid_root / 'better_null_results.csv'}")


def _write_results(out_dir: Path, rows: list[dict], pair_rows: list[dict]) -> None:
    result_fields = [
        "target",
        "rpn_bbox_loss",
        "rpn_bbox_direction",
        "rpn_obj_loss",
        "rpn_obj_direction",
        "roi_bbox_loss",
        "roi_bbox_direction",
        "roi_cls_loss",
        "roi_cls_direction",
        "layer_grad_run_dir",
        "meta_classifier_run_dir",
        "auroc",
        "ap",
        "ece",
        "ace",
    ]
    _write_csv(out_dir / "grid_results.csv", rows, result_fields)

    pair_rows = sorted(pair_rows, key=lambda r: (r["delta_auroc"], r["delta_ap"]), reverse=True)
    pair_fields = [
        "rpn_bbox_loss",
        "rpn_bbox_direction",
        "rpn_obj_loss",
        "rpn_obj_direction",
        "roi_bbox_loss",
        "roi_bbox_direction",
        "roi_cls_loss",
        "roi_cls_direction",
        "cand_auroc",
        "null_auroc",
        "delta_auroc",
        "cand_ap",
        "null_ap",
        "delta_ap",
        "null_better_auroc",
        "null_better_ap",
        "cand_layer_grad_run_dir",
        "null_layer_grad_run_dir",
        "cand_meta_classifier_run_dir",
        "null_meta_classifier_run_dir",
    ]
    _write_csv(out_dir / "target_pair_comparison.csv", pair_rows, pair_fields)
    _write_csv(
        out_dir / "better_null_results.csv",
        [row for row in pair_rows if row["null_better_auroc"]],
        pair_fields,
    )


if __name__ == "__main__":
    main()
