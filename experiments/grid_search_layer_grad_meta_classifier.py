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
OBJECT_DETECTOR_CONFIG = r"object_detectors/configs/predict_coco_yolov5.yaml"
META_CLASSIFIER_CONFIG = (
    r"meta_models/meta_classifier/configs/train_meta_classifier.yaml"
)

# If empty, dataset.gt_root is read from META_CLASSIFIER_CONFIG.
GT_ROOT = ""

# Set to a fixed name to resume a previous grid root. Empty string creates a new timestamped root.
GRID_NAME = ""

RUN_LAYER_GRAD = True
RUN_META_CLASSIFIER = True
REUSE_EXISTING = True

# Use None for all valid combinations, or set a small int for a smoke test.
MAX_COMBINATIONS = None


TARGETS = ["cand_target", "null_target"]
BBOX_LOSSES = ["ciou", "l1", "l2"]
BBOX_DIRECTIONS = ["pred_to_target", "target_to_pred"]
CLS_LOSSES = ["bcewithlogits", "kl", "ce"]
OBJ_LOSSES = ["bcewithlogits", "abs_diff", "signed_diff"]


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


def _combo_slug(combo: dict) -> str:
    return (
        f"layer_grad_t-{_abbr(combo['target'])}"
        f"__b-{combo['bbox_loss']}-{_abbr(combo['bbox_direction'])}"
        f"__c-{_abbr(combo['cls_loss'])}-{_abbr(combo['cls_direction'])}"
        f"__o-{_abbr(combo['obj_loss'])}-{_abbr(combo['obj_direction'])}"
    )


def _timestamped_combo_dir(root: Path, slug: str) -> Path:
    existing = sorted(root.glob(f"??-??-????_??;??_{slug}"))
    if existing and REUSE_EXISTING:
        return existing[-1]
    return root / f"{datetime.now().strftime('%m-%d-%Y_%H;%M')}_{slug}"


def _valid_cls_directions(cls_loss: str) -> list[str]:
    if cls_loss == "kl":
        return ["pred_to_target", "target_to_pred"]
    return ["pred_to_target"]


def _valid_bbox_directions(_bbox_loss: str) -> list[str]:
    return ["pred_to_target"]


def _valid_obj_directions(obj_loss: str) -> list[str]:
    if obj_loss in {"bcewithlogits", "abs_diff"}:
        return ["pred_to_target"]
    return ["pred_to_target", "target_to_pred"]


def iter_combinations():
    count = 0
    for bbox_loss, cls_loss, obj_loss in itertools.product(
        BBOX_LOSSES,
        CLS_LOSSES,
        OBJ_LOSSES,
    ):
        for bbox_direction in _valid_bbox_directions(bbox_loss):
            for cls_direction in _valid_cls_directions(cls_loss):
                for obj_direction in _valid_obj_directions(obj_loss):
                    for target in TARGETS:
                        yield {
                            "target": target,
                            "bbox_loss": bbox_loss,
                            "bbox_direction": bbox_direction,
                            "cls_loss": cls_loss,
                            "cls_direction": cls_direction,
                            "obj_loss": obj_loss,
                            "obj_direction": obj_direction,
                        }
                        count += 1
                        if MAX_COMBINATIONS is not None and count >= int(MAX_COMBINATIONS):
                            return


def _run(cmd: list[str]) -> None:
    print(" ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _prepare_layer_grad_config(base_config: dict, combo: dict) -> dict:
    config = deepcopy(base_config)
    config["mode"] = "predict"
    output_cfg = config.setdefault("output", {})
    output_cfg["uncertainty"] = "layer_grad"
    layer_grad_cfg = output_cfg.setdefault("layer_grad", {})
    layer_grad_cfg.setdefault("save_csv", {})["enabled"] = True
    grad_cfg = layer_grad_cfg.setdefault("gradient", {})
    grad_cfg.update(
        {
            "target": combo["target"],
            "bbox_loss": combo["bbox_loss"],
            "cls_loss": combo["cls_loss"],
            "obj_loss": combo["obj_loss"],
            "bbox_direction": combo["bbox_direction"],
            "cls_direction": combo["cls_direction"],
            "obj_direction": combo["obj_direction"],
        }
    )
    return config


def _prepare_meta_config(
    base_config: dict, layer_grad_run_dir: Path, gt_root: str
) -> dict:
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
        reader = csv.DictReader(f)
        rows = list(reader)
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


def _comparison_key(row: dict) -> tuple:
    return (
        row["bbox_loss"],
        row["bbox_direction"],
        row["cls_loss"],
        row["cls_direction"],
        row["obj_loss"],
        row["obj_direction"],
    )


def main() -> None:
    od_config_path = _resolve_path(OBJECT_DETECTOR_CONFIG)
    meta_config_path = _resolve_path(META_CLASSIFIER_CONFIG)
    od_base_config = _load_yaml(od_config_path)
    meta_base_config = _load_yaml(meta_config_path)
    dataset = _dataset_name(od_base_config)

    grid_name = (
        GRID_NAME.strip()
        or f"{datetime.now().strftime('%m-%d-%Y_%H;%M')}_layer_grad_grid"
    )
    od_grid_root = (
        REPO_ROOT / "object_detectors" / "runs" / "predict" / dataset / grid_name
    )
    meta_grid_root = (
        REPO_ROOT
        / "meta_models"
        / "meta_classifier"
        / "runs"
        / "train"
        / dataset
        / grid_name
    )
    od_grid_root.mkdir(parents=True, exist_ok=True)
    meta_grid_root.mkdir(parents=True, exist_ok=True)

    gt_root = (
        GT_ROOT.strip()
        or str(meta_base_config.get("dataset", {}).get("gt_root", "")).strip()
    )
    if not gt_root:
        raise ValueError(
            "GT_ROOT is empty and META_CLASSIFIER_CONFIG has no dataset.gt_root."
        )

    rows = []
    combo_list = list(iter_combinations())
    for idx, combo in enumerate(combo_list, start=1):
        slug = _combo_slug(combo)
        print(f"[{idx}/{len(combo_list)}] {slug}", flush=True)

        layer_dir = _timestamped_combo_dir(od_grid_root, slug)
        meta_dir = _timestamped_combo_dir(meta_grid_root, slug)
        layer_csv = layer_dir / "layer_grad.csv"
        eval_csv = meta_dir / "results" / "evaluation_results.csv"

        if RUN_LAYER_GRAD and not (REUSE_EXISTING and layer_csv.is_file()):
            od_config = _prepare_layer_grad_config(od_base_config, combo)
            od_config_path_i = layer_dir / "grid_object_detector_config.yaml"
            _save_yaml(od_config, od_config_path_i)
            _run(
                [
                    sys.executable,
                    "object_detectors/main.py",
                    "--config",
                    str(od_config_path_i),
                    "--run-dir",
                    str(layer_dir),
                ]
            )

        if RUN_META_CLASSIFIER and not (REUSE_EXISTING and eval_csv.is_file()):
            meta_config = _prepare_meta_config(meta_base_config, layer_dir, gt_root)
            meta_config_path_i = meta_dir / "grid_meta_classifier_config.yaml"
            _save_yaml(meta_config, meta_config_path_i)
            _run(
                [
                    sys.executable,
                    "meta_models/meta_classifier/main.py",
                    "--config",
                    str(meta_config_path_i),
                    "--run-dir",
                    str(meta_dir),
                ]
            )

        metrics = _read_mean_metrics(eval_csv)
        rows.append(
            {
                **combo,
                "layer_grad_run_dir": str(layer_dir),
                "meta_classifier_run_dir": str(meta_dir),
                "auroc": metrics.get("auroc", ""),
                "ap": metrics.get("ap", ""),
                "ece": metrics.get("ece", ""),
                "ace": metrics.get("ace", ""),
            }
        )
        _write_results(meta_grid_root, rows)

    print(f"Saved grid results: {meta_grid_root / 'grid_results.csv'}")
    print(f"Saved pair comparison: {meta_grid_root / 'target_pair_comparison.csv'}")
    print(f"Saved null comparison: {meta_grid_root / 'better_null_results.csv'}")


def _write_results(out_dir: Path, rows: list[dict]) -> None:
    fields = [
        "target",
        "bbox_loss",
        "bbox_direction",
        "cls_loss",
        "cls_direction",
        "obj_loss",
        "obj_direction",
        "layer_grad_run_dir",
        "meta_classifier_run_dir",
        "auroc",
        "ap",
        "ece",
        "ace",
    ]
    _write_csv(out_dir / "grid_results.csv", rows, fields)

    by_key = {}
    for row in rows:
        by_key.setdefault(_comparison_key(row), {})[row["target"]] = row

    compare_rows = []
    for key, pair in by_key.items():
        if "cand_target" not in pair or "null_target" not in pair:
            continue
        cand = pair["cand_target"]
        null = pair["null_target"]
        try:
            cand_auroc = float(cand["auroc"])
            null_auroc = float(null["auroc"])
            cand_ap = float(cand["ap"])
            null_ap = float(null["ap"])
        except Exception:
            continue
        compare_rows.append(
            {
                "bbox_loss": key[0],
                "bbox_direction": key[1],
                "cls_loss": key[2],
                "cls_direction": key[3],
                "obj_loss": key[4],
                "obj_direction": key[5],
                "cand_auroc": cand_auroc,
                "null_auroc": null_auroc,
                "delta_auroc": null_auroc - cand_auroc,
                "cand_ap": cand_ap,
                "null_ap": null_ap,
                "delta_ap": null_ap - cand_ap,
                "null_better_auroc": int(null_auroc > cand_auroc),
                "null_better_ap": int(null_ap > cand_ap),
            }
        )
    compare_rows.sort(key=lambda r: (r["delta_auroc"], r["delta_ap"]), reverse=True)
    comparison_fields = [
        "bbox_loss",
        "bbox_direction",
        "cls_loss",
        "cls_direction",
        "obj_loss",
        "obj_direction",
        "cand_auroc",
        "null_auroc",
        "delta_auroc",
        "cand_ap",
        "null_ap",
        "delta_ap",
        "null_better_auroc",
        "null_better_ap",
    ]
    _write_csv(
        out_dir / "target_pair_comparison.csv",
        compare_rows,
        comparison_fields,
    )
    _write_csv(
        out_dir / "better_null_results.csv",
        [row for row in compare_rows if row["null_better_auroc"]],
        comparison_fields,
    )


if __name__ == "__main__":
    main()
