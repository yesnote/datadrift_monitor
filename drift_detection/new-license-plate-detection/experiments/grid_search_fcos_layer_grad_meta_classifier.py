import csv
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
OBJECT_DETECTORS_ROOT = REPO_ROOT / "object_detectors"
if str(OBJECT_DETECTORS_ROOT) not in sys.path:
    sys.path.insert(0, str(OBJECT_DETECTORS_ROOT))

from commands.predict.common import (  # noqa: E402
    StageTimingProfiler,
    _as_image_list,
    _prepare_infer_batch,
)
from commands.predict.fcos.common import select_fcos_post_nms  # noqa: E402
from commands.predict.fcos.layer_grad import (  # noqa: E402
    _build_fcos_losses,
    _gradient_to_np_array,
    _parse_fcos_layer_grad_config,
    _prediction_class_name,
    _resolve_fcos_candidate_sources,
    _safe_npz_key,
)
from commands.utils.predict_utils import (  # noqa: E402
    build_detector,
    expand_layer_names,
    format_gradient_output,
    parse_output_config,
    resolve_layer_parameter,
)
from dataloaders.dataloader_yolo import create_dataloader  # noqa: E402

# Edit these paths before running.
OBJECT_DETECTOR_CONFIG = r"object_detectors/configs/fcos/predict_coco_fcos.yaml"
META_CLASSIFIER_CONFIG = (
    r"meta_models/meta_classifier/configs/train_meta_classifier.yaml"
)

# If empty, dataset.gt_root is read from META_CLASSIFIER_CONFIG.
GT_ROOT = ""

# Set to a fixed name to resume a previous grid root. Empty string creates a new timestamped root.
GRID_NAME = ""

RUN_LAYER_GRAD = True
RUN_META_CLASSIFIER = True
REUSE_EXISTING = False

# Use None for all meta-classifier combinations, or set a small int for a smoke test.
# Object detector term CSVs are still generated once for all 18 single-term settings.
MAX_COMBINATIONS = None

TARGETS = ["cand_target", "null_target"]
BBOX_LOSSES = ["l1", "l2"]
CLS_LOSSES = ["bcewithlogits", "kl"]
CNT_LOSSES = ["bcewithlogits", "abs_diff", "signed_diff"]


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
    raw = str(config.get("model", {}).get("type", "fcos")).strip().lower()
    return (
        "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in raw).strip(
            "_"
        )
        or "fcos"
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
        "bbox_loss": "bbox",
        "cls_loss": "cls",
        "cnt_loss": "cnt",
    }
    return aliases.get(value, value)


def _valid_cls_directions(cls_loss: str) -> list[str]:
    if cls_loss == "kl":
        return ["pred_to_target", "target_to_pred"]
    return ["pred_to_target"]


def _valid_cnt_directions(cnt_loss: str) -> list[str]:
    if cnt_loss == "signed_diff":
        return ["pred_to_target", "target_to_pred"]
    return ["pred_to_target"]


def _combo_slug(combo: dict) -> str:
    term = combo["term"]
    if term == "bbox_loss":
        spec = f"b-{combo['bbox_loss']}-{_abbr(combo['bbox_direction'])}"
    elif term == "cls_loss":
        spec = f"c-{_abbr(combo['cls_loss'])}-{_abbr(combo['cls_direction'])}"
    elif term == "cnt_loss":
        spec = f"cnt-{_abbr(combo['cnt_loss'])}-{_abbr(combo['cnt_direction'])}"
    else:
        raise ValueError(f"Unsupported FCOS layer_grad term: {term}")
    return f"layer_grad_t-{_abbr(combo['target'])}__term-{_abbr(term)}__{spec}"


def _timestamped_combo_dir(root: Path, slug: str) -> Path:
    existing = sorted(root.glob(f"??-??-????_??;??_{slug}"))
    if existing and REUSE_EXISTING:
        return existing[-1]
    return root / f"{datetime.now().strftime('%m-%d-%Y_%H;%M')}_{slug}"


def _timestamped_meta_dir(root: Path, index: int, target: str) -> Path:
    suffix = f"m{int(index):03d}_{_abbr(target)}"
    existing = sorted(root.glob(f"??-??-????_??;??_{suffix}"))
    if existing and REUSE_EXISTING:
        return existing[-1]
    return root / f"{datetime.now().strftime('%m-%d-%Y_%H;%M')}_{suffix}"


def _base_combo(target: str, term: str) -> dict:
    return {
        "target": target,
        "term": term,
        "bbox_loss": "l1",
        "bbox_direction": "pred_to_target",
        "cls_loss": "bcewithlogits",
        "cls_direction": "pred_to_target",
        "cnt_loss": "bcewithlogits",
        "cnt_direction": "pred_to_target",
    }


def iter_term_combinations():
    for target in TARGETS:
        for bbox_loss in BBOX_LOSSES:
            combo = _base_combo(target, "bbox_loss")
            combo.update({"bbox_loss": bbox_loss})
            yield combo

        for cls_loss in CLS_LOSSES:
            for cls_direction in _valid_cls_directions(cls_loss):
                combo = _base_combo(target, "cls_loss")
                combo.update({"cls_loss": cls_loss, "cls_direction": cls_direction})
                yield combo

        for cnt_loss in CNT_LOSSES:
            for cnt_direction in _valid_cnt_directions(cnt_loss):
                combo = _base_combo(target, "cnt_loss")
                combo.update({"cnt_loss": cnt_loss, "cnt_direction": cnt_direction})
                yield combo


def _term_combo_key(combo: dict) -> tuple:
    return (
        combo["target"],
        combo["term"],
        combo["bbox_loss"],
        combo["bbox_direction"],
        combo["cls_loss"],
        combo["cls_direction"],
        combo["cnt_loss"],
        combo["cnt_direction"],
    )


def _meta_combo_slug(combo: dict) -> str:
    return (
        f"layer_grad_t-{_abbr(combo['target'])}"
        f"__b-{combo['bbox_loss']}-{_abbr(combo['bbox_direction'])}"
        f"__c-{_abbr(combo['cls_loss'])}-{_abbr(combo['cls_direction'])}"
        f"__cnt-{_abbr(combo['cnt_loss'])}-{_abbr(combo['cnt_direction'])}"
    )


def iter_meta_combinations(term_run_dirs: dict[tuple, Path]):
    count = 0
    term_combos = list(iter_term_combinations())
    for target in TARGETS:
        bbox_combos = [
            combo
            for combo in term_combos
            if combo["target"] == target and combo["term"] == "bbox_loss"
        ]
        cls_combos = [
            combo
            for combo in term_combos
            if combo["target"] == target and combo["term"] == "cls_loss"
        ]
        cnt_combos = [
            combo
            for combo in term_combos
            if combo["target"] == target and combo["term"] == "cnt_loss"
        ]
        for bbox_combo in bbox_combos:
            for cls_combo in cls_combos:
                for cnt_combo in cnt_combos:
                    yield {
                        "target": target,
                        "bbox_loss": bbox_combo["bbox_loss"],
                        "bbox_direction": bbox_combo["bbox_direction"],
                        "cls_loss": cls_combo["cls_loss"],
                        "cls_direction": cls_combo["cls_direction"],
                        "cnt_loss": cnt_combo["cnt_loss"],
                        "cnt_direction": cnt_combo["cnt_direction"],
                        "input_roots": [
                            term_run_dirs[_term_combo_key(bbox_combo)],
                            term_run_dirs[_term_combo_key(cls_combo)],
                            term_run_dirs[_term_combo_key(cnt_combo)],
                        ],
                    }
                    count += 1
                    if MAX_COMBINATIONS is not None and count >= int(MAX_COMBINATIONS):
                        return


def _run(cmd: list[str]) -> None:
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
            "scalar": [combo["term"]],
            "bbox_loss": combo["bbox_loss"],
            "cls_loss": combo["cls_loss"],
            "cnt_loss": combo["cnt_loss"],
            "bbox_direction": combo["bbox_direction"],
            "cls_direction": combo["cls_direction"],
            "cnt_direction": combo["cnt_direction"],
        }
    )
    return config


def _prepare_meta_config(
    base_config: dict, input_roots: list[Path], gt_root: str
) -> dict:
    config = deepcopy(base_config)
    config["mode"] = "train"
    dataset_cfg = config.setdefault("dataset", {})
    dataset_cfg["input_root"] = [str(path) for path in input_roots]
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


def _timing_stages_for_target(target: str) -> list[str]:
    stages = [
        "detector_inference_sec",
        "loss_compute_sec",
        "backpropagation_sec",
        "feature_compute_sec",
    ]
    if target == "cand_target":
        stages.insert(1, "candidate_search_sec")
    return stages


def _empty_stage_seconds(target: str, detector_inference_sec: float) -> dict:
    stages = {
        "detector_inference_sec": float(detector_inference_sec),
        "loss_compute_sec": 0.0,
        "backpropagation_sec": 0.0,
        "feature_compute_sec": 0.0,
    }
    if target == "cand_target":
        stages["candidate_search_sec"] = 0.0
    return stages


def _layer_grad_fieldnames(combo: dict, target_layers: list[str], reductions: list[str]) -> list[str]:
    fieldnames = [
        "image_id", "image_path", "pred_idx", "raw_pred_idx",
        "xmin", "ymin", "xmax", "ymax", "score", "pred_class",
    ]
    save_raw_gradients = not reductions
    for layer_name in target_layers:
        grad_key = f"{combo['term']}_{layer_name}"
        if save_raw_gradients:
            fieldnames.append(grad_key)
        else:
            fieldnames.extend(f"{grad_key}_{metric}" for metric in reductions)
    return fieldnames


def _run_fcos_layer_grad_terms_once(od_base_config: dict, combo_run_dirs: dict[tuple, Path]) -> None:
    if not RUN_LAYER_GRAD:
        return

    active = []
    for combo in iter_term_combinations():
        key = _term_combo_key(combo)
        run_dir = combo_run_dirs[key]
        if REUSE_EXISTING and (run_dir / "layer_grad.csv").is_file():
            continue
        active.append((key, combo, run_dir))
    if not active:
        return

    first_config = _prepare_layer_grad_config(od_base_config, active[0][1])
    dataset_cfg = first_config.get("dataset", {})
    split = dataset_cfg.get("split", "val")
    parsed_common = parse_output_config(first_config.get("output", {}))
    common_layer_cfg = _parse_fcos_layer_grad_config(first_config)

    dataloader = create_dataloader(first_config, split=split)
    if len(dataloader.dataset) == 0:
        raise ValueError("Loaded 0 images. Check dataset root/image_dir/split configuration in YAML.")

    detector, device = build_detector(first_config)
    if not bool(getattr(detector, "is_fcos", False)):
        raise ValueError("grid_search_fcos_layer_grad_meta_classifier.py requires model.type=fcos.")

    reductions = common_layer_cfg["reduction"]
    save_raw_gradients = not reductions
    handles = {}
    profilers = {}
    target_layers_by_key = {}
    layer_params_by_key = {}
    raw_gradient_dirs = {}
    original_requires_grad = {}

    try:
        for key, combo, run_dir in active:
            run_dir.mkdir(parents=True, exist_ok=True)
            od_config = _prepare_layer_grad_config(od_base_config, combo)
            _save_yaml(od_config, run_dir / "grid_object_detector_config.yaml")
            layer_cfg = _parse_fcos_layer_grad_config(od_config)
            target_layers = expand_layer_names(
                detector,
                layer_cfg["target_layer_map"].get(combo["term"], []),
            )
            layer_params = [resolve_layer_parameter(detector, name) for name in target_layers]
            for param in layer_params:
                original_requires_grad.setdefault(id(param), (param, bool(param.requires_grad)))
                param.requires_grad_(True)

            csv_file = open(run_dir / "layer_grad.csv", "w", newline="", encoding="utf-8")
            writer = csv.DictWriter(
                csv_file,
                fieldnames=_layer_grad_fieldnames(combo, target_layers, reductions),
            )
            writer.writeheader()
            handles[key] = (combo, run_dir, csv_file, writer, layer_cfg)
            target_layers_by_key[key] = target_layers
            layer_params_by_key[key] = layer_params
            profilers[key] = StageTimingProfiler(
                run_dir=run_dir,
                uncertainty="layer_grad",
                unit=parsed_common.get("unit", "bbox"),
                stages=_timing_stages_for_target(combo["target"]),
                device=device,
            )
            if save_raw_gradients:
                raw_dir = run_dir / "gradients"
                raw_dir.mkdir(parents=True, exist_ok=True)
                raw_gradient_dirs[key] = raw_dir

        for batch_idx, (images, targets) in enumerate(tqdm(
            dataloader,
            desc="Object Detector (predict - layer_grad grid)",
            total=len(dataloader),
        )):
            image_list = _as_image_list(images)
            infer_batch, _ratios, _pads, _resized_chws = _prepare_infer_batch(
                detector, image_list, device, auto=False
            )
            fcos_preprocessed = detector.preprocess_images(infer_batch)

            detector.zero_grad(set_to_none=True)
            t_detector = profilers[next(iter(profilers))].start()
            pre_nms_threshold = float(getattr(detector, "confidence", 0.05))
            if any(combo["target"] == "cand_target" for combo, *_ in handles.values()):
                pre_nms_threshold = min(
                    pre_nms_threshold,
                    float(common_layer_cfg["cand_score_threshold"]),
                )
            with detector.temporary_pre_nms_threshold(pre_nms_threshold):
                model_output = detector.forward_layer_grad(fcos_preprocessed)
            selected_preds, _selected_logits, _selected_objectness, selected_indices = select_fcos_post_nms(
                detector,
                model_output["post_prediction"],
                model_output["post_logits"],
                model_output["post_indices"],
                conf_thres=float(getattr(detector, "confidence", getattr(detector, "conf_thresh", 0.05))),
            )
            detector_inference_sec = profilers[next(iter(profilers))].elapsed(t_detector)

            stage_by_key = {
                key: _empty_stage_seconds(combo["target"], detector_inference_sec)
                for key, (combo, _run_dir, _csv_file, _writer, _layer_cfg) in handles.items()
            }
            rows_by_key = {key: [] for key in handles}
            grad_arrays_by_key = {key: {} for key in handles}
            batch_items = 0

            cand_keys = [
                key for key, (combo, *_rest) in handles.items()
                if combo["target"] == "cand_target"
            ]
            null_keys = [
                key for key, (combo, *_rest) in handles.items()
                if combo["target"] == "null_target"
            ]

            for sample_idx in range(len(image_list)):
                target = targets[sample_idx]
                image_id = int(target["image_id"][0].item())
                image_path = target["path"]
                det = (
                    selected_preds[sample_idx]
                    if selected_preds and sample_idx < len(selected_preds)
                    else torch.zeros((0, 6), device=device)
                )
                raw_keep = (
                    selected_indices[sample_idx]
                    if selected_indices and sample_idx < len(selected_indices)
                    else torch.zeros((0,), dtype=torch.long, device=device)
                )
                batch_items += int(det.shape[0])

                for pred_idx in range(int(det.shape[0])):
                    raw_idx = int(raw_keep[pred_idx].detach().cpu().item()) if pred_idx < int(raw_keep.shape[0]) else pred_idx
                    final_box = det[pred_idx, :4]
                    final_cls = int(det[pred_idx, 5].detach().cpu().item()) if det.shape[1] > 5 else 0
                    base_row = {
                        "image_id": image_id,
                        "image_path": image_path,
                        "pred_idx": pred_idx,
                        "raw_pred_idx": raw_idx,
                        "xmin": float(det[pred_idx, 0].detach().cpu().item()),
                        "ymin": float(det[pred_idx, 1].detach().cpu().item()),
                        "xmax": float(det[pred_idx, 2].detach().cpu().item()),
                        "ymax": float(det[pred_idx, 3].detach().cpu().item()),
                        "score": float(det[pred_idx, 4].detach().cpu().item()),
                        "pred_class": _prediction_class_name(detector, final_cls),
                    }

                    cand_sources = None
                    if cand_keys:
                        candidate_timing = {"candidate_search_sec": 0.0}
                        cand_sources = _resolve_fcos_candidate_sources(
                            target_mode="cand_target",
                            model_output=model_output,
                            image_idx=sample_idx,
                            pred_idx=pred_idx,
                            final_box=final_box,
                            final_cls=final_cls,
                            raw_idx=raw_idx,
                            cand_score_threshold=common_layer_cfg["cand_score_threshold"],
                            timing=profilers[cand_keys[0]],
                            timing_accumulator=candidate_timing,
                        )
                        for cand_key in cand_keys:
                            stage_by_key[cand_key]["candidate_search_sec"] += candidate_timing["candidate_search_sec"]

                    null_sources = None
                    if null_keys:
                        null_sources = _resolve_fcos_candidate_sources(
                            target_mode="null_target",
                            model_output=model_output,
                            image_idx=sample_idx,
                            pred_idx=pred_idx,
                            final_box=final_box,
                            final_cls=final_cls,
                            raw_idx=raw_idx,
                            cand_score_threshold=common_layer_cfg["cand_score_threshold"],
                            timing=profilers[null_keys[0]],
                            timing_accumulator={"candidate_search_sec": 0.0},
                        )

                    for key, (combo, _run_dir, _csv_file, _writer, layer_cfg) in handles.items():
                        detector.zero_grad(set_to_none=True)
                        sources = cand_sources if combo["target"] == "cand_target" else null_sources
                        losses = _build_fcos_losses(
                            target_mode=combo["target"],
                            target_values=[combo["term"]],
                            model_output=model_output,
                            image_idx=sample_idx,
                            pred_idx=pred_idx,
                            final_box=final_box,
                            final_cls=final_cls,
                            raw_idx=raw_idx,
                            cand_score_threshold=layer_cfg["cand_score_threshold"],
                            bbox_loss=combo["bbox_loss"],
                            cls_loss=combo["cls_loss"],
                            cnt_loss=combo["cnt_loss"],
                            bbox_direction=combo["bbox_direction"],
                            cls_direction=combo["cls_direction"],
                            cnt_direction=combo["cnt_direction"],
                            timing=profilers[key],
                            timing_accumulator=stage_by_key[key],
                            candidate_sources=sources,
                        )
                        scalar = losses.get(combo["term"])
                        row = dict(base_row)
                        target_layers = target_layers_by_key[key]
                        layer_params = layer_params_by_key[key]
                        if scalar is None:
                            for layer_name in target_layers:
                                grad_key = f"{combo['term']}_{layer_name}"
                                if save_raw_gradients:
                                    row[grad_key] = ""
                                else:
                                    for metric in reductions:
                                        row[f"{grad_key}_{metric}"] = 0.0
                            rows_by_key[key].append(row)
                            continue

                        t_backprop = profilers[key].start()
                        grads = torch.autograd.grad(
                            scalar,
                            layer_params,
                            retain_graph=True,
                            allow_unused=True,
                        )
                        stage_by_key[key]["backpropagation_sec"] += profilers[key].elapsed(t_backprop)

                        t_feature = profilers[key].start()
                        for layer_idx, layer_name in enumerate(target_layers):
                            grad_key = f"{combo['term']}_{layer_name}"
                            grad_value = format_gradient_output(
                                grads[layer_idx],
                                vector_reduction=reductions,
                                map_reduction="none",
                            )
                            if save_raw_gradients:
                                array_key = f"r{len(grad_arrays_by_key[key]):06d}_{_safe_npz_key(grad_key)}"
                                grad_arrays_by_key[key][array_key] = _gradient_to_np_array(grad_value)
                                row[grad_key] = f"gradients/layer_grad_batch_{batch_idx:06d}.npz::{array_key}"
                            else:
                                for metric in reductions:
                                    value = grad_value.get(metric, 0.0) if isinstance(grad_value, dict) else 0.0
                                    row[f"{grad_key}_{metric}"] = (
                                        float(value.detach().cpu().item())
                                        if isinstance(value, torch.Tensor)
                                        else float(value)
                                    )
                        stage_by_key[key]["feature_compute_sec"] += profilers[key].elapsed(t_feature)
                        rows_by_key[key].append(row)
                        del scalar, grads

            for key, (combo, run_dir, csv_file, writer, _layer_cfg) in handles.items():
                if save_raw_gradients and grad_arrays_by_key[key]:
                    np.savez(
                        raw_gradient_dirs[key] / f"layer_grad_batch_{batch_idx:06d}.npz",
                        **grad_arrays_by_key[key],
                    )
                writer.writerows(rows_by_key[key])
                csv_file.flush()
                profilers[key].record(
                    num_images=len(image_list),
                    num_predictions=batch_items,
                    stage_seconds=stage_by_key[key],
                )

            del infer_batch, fcos_preprocessed, model_output, selected_preds, selected_indices
            detector.zero_grad(set_to_none=True)
    finally:
        for param, req_grad in original_requires_grad.values():
            param.requires_grad_(req_grad)
        for key, (_combo, _run_dir, csv_file, _writer, _layer_cfg) in handles.items():
            csv_file.close()
            profilers[key].save()
        del detector
        if "device" in locals() and device.type == "cuda":
            torch.cuda.empty_cache()


def _run_layer_grad_if_needed(
    od_base_config: dict,
    od_grid_root: Path,
    combo: dict,
) -> Path:
    slug = _combo_slug(combo)
    layer_dir = _timestamped_combo_dir(od_grid_root, slug)
    layer_csv = layer_dir / "layer_grad.csv"
    if RUN_LAYER_GRAD and not (REUSE_EXISTING and layer_csv.is_file()):
        od_config = _prepare_layer_grad_config(od_base_config, combo)
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


def _result_row(combo: dict, metrics: dict) -> dict:
    return {
        "target": combo["target"],
        "bbox_loss": combo["bbox_loss"],
        "bbox_direction": combo["bbox_direction"],
        "cls_loss": combo["cls_loss"],
        "cls_direction": combo["cls_direction"],
        "cnt_loss": combo["cnt_loss"],
        "cnt_direction": combo["cnt_direction"],
        "auroc": metrics.get("auroc", ""),
        "ap": metrics.get("ap", ""),
        "ece": metrics.get("ece", ""),
        "ace": metrics.get("ace", ""),
    }


def _comparison_key(row: dict) -> tuple:
    return (
        row["bbox_loss"],
        row["bbox_direction"],
        row["cls_loss"],
        row["cls_direction"],
        row["cnt_loss"],
        row["cnt_direction"],
    )


def _write_results(out_dir: Path, rows: list[dict]) -> None:
    result_fields = [
        "target",
        "bbox_loss",
        "bbox_direction",
        "cls_loss",
        "cls_direction",
        "cnt_loss",
        "cnt_direction",
        "auroc",
        "ap",
        "ece",
        "ace",
    ]
    _write_csv(out_dir / "grid_results.csv", rows, result_fields)

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
                "cnt_loss": key[4],
                "cnt_direction": key[5],
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
        "cnt_loss",
        "cnt_direction",
        "cand_auroc",
        "null_auroc",
        "delta_auroc",
        "cand_ap",
        "null_ap",
        "delta_ap",
        "null_better_auroc",
        "null_better_ap",
    ]
    _write_csv(out_dir / "target_pair_comparison.csv", compare_rows, comparison_fields)
    _write_csv(
        out_dir / "better_null_results.csv",
        [row for row in compare_rows if row["null_better_auroc"]],
        comparison_fields,
    )


def main() -> None:
    od_config_path = _resolve_path(OBJECT_DETECTOR_CONFIG)
    meta_config_path = _resolve_path(META_CLASSIFIER_CONFIG)
    od_base_config = _load_yaml(od_config_path)
    meta_base_config = _load_yaml(meta_config_path)
    dataset = _dataset_name(od_base_config)
    model = _model_name(od_base_config)

    grid_name = (
        GRID_NAME.strip()
        or f"{datetime.now().strftime('%m-%d-%Y_%H;%M')}_fcos_layer_grad_grid"
    )
    od_grid_root = (
        REPO_ROOT
        / "object_detectors"
        / "runs"
        / model
        / "predict"
        / dataset
        / grid_name
    )
    meta_grid_root = (
        REPO_ROOT
        / "meta_models"
        / "meta_classifier"
        / "runs"
        / model
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

    term_run_dirs: dict[tuple, Path] = {}
    term_combo_list = list(iter_term_combinations())
    print(f"Layer-grad term runs: {len(term_combo_list)}", flush=True)
    for idx, combo in enumerate(term_combo_list, start=1):
        slug = _combo_slug(combo)
        print(f"[OD {idx}/{len(term_combo_list)}] {slug}", flush=True)
        layer_dir = _timestamped_combo_dir(od_grid_root, slug)
        term_run_dirs[_term_combo_key(combo)] = layer_dir
    _run_fcos_layer_grad_terms_once(od_base_config, term_run_dirs)

    rows = []
    meta_combo_list = list(iter_meta_combinations(term_run_dirs))
    cand_count = sum(1 for combo in meta_combo_list if combo["target"] == "cand_target")
    null_count = sum(1 for combo in meta_combo_list if combo["target"] == "null_target")
    print(
        f"Meta-classifier combinations: {len(meta_combo_list)} "
        f"(cand_target={cand_count}, null_target={null_count})",
        flush=True,
    )
    for idx, combo in enumerate(meta_combo_list, start=1):
        slug = _meta_combo_slug(combo)
        print(f"[META {idx}/{len(meta_combo_list)}] {slug}", flush=True)

        meta_dir = _timestamped_meta_dir(meta_grid_root, idx, combo["target"])
        eval_csv = meta_dir / "results" / "evaluation_results.csv"
        if RUN_META_CLASSIFIER and not (REUSE_EXISTING and eval_csv.is_file()):
            meta_dir.mkdir(parents=True, exist_ok=True)
            meta_config = _prepare_meta_config(
                meta_base_config, combo["input_roots"], gt_root
            )
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

        metrics = _read_mean_metrics(eval_csv)
        rows.append(_result_row(combo, metrics))
        _write_results(meta_grid_root, rows)

    print(f"Saved grid results: {meta_grid_root / 'grid_results.csv'}")
    print(f"Saved pair comparison: {meta_grid_root / 'target_pair_comparison.csv'}")
    print(f"Saved null comparison: {meta_grid_root / 'better_null_results.csv'}")


if __name__ == "__main__":
    main()
