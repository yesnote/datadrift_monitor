import argparse
import re
import sys
import warnings
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from error_detectors.commands.run_train import run_train
from error_detectors.commands.utils.run_utils import create_run_dir, save_used_config


def resolve_config_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path_value(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def normalize_input_roots(raw_value) -> list[str]:
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
        for cue_name in ("feature_grad", "layer_grad", "full_softmax", "mc_dropout", "entropy", "energy", "feature", "score", "gt", "fn", "tp"):
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="error_detectors/configs/experiment_error_detector.yaml")
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = load_config(config_path)

    dataset_cfg = config.get("dataset", {})
    input_root_raw_list = normalize_input_roots(dataset_cfg.get("input_root", ""))
    gt_root_raw = str(dataset_cfg.get("gt_root", "")).strip()
    if not input_root_raw_list or not gt_root_raw:
        raise ValueError("dataset.input_root (str or list[str]) and dataset.gt_root are required.")

    input_roots = [resolve_path_value(v) for v in input_root_raw_list]
    gt_root = resolve_path_value(gt_root_raw)
    parsed_inputs = [parse_root_info(p) for p in input_roots]
    input_groups = {group for group, _cue, _target in parsed_inputs}
    if len(input_groups) != 1:
        msg = f"All dataset.input_root entries must share one model group, got: {sorted(input_groups)}"
        warnings.warn(msg)
        raise ValueError(msg)
    input_group = next(iter(input_groups))
    cue_parts = [cue for _group, cue, _target in parsed_inputs]
    target_parts = [target for _group, _cue, target in parsed_inputs if target]
    input_cue = "+".join(cue_parts)
    input_target = "+".join(target_parts)
    gt_group, _gt_cue, _gt_target = parse_root_info(gt_root)
    if input_group != gt_group:
        msg = (
            "dataset.input_root and dataset.gt_root must have the same model group "
            f"(got '{input_group}' vs '{gt_group}')."
        )
        warnings.warn(msg)
        raise ValueError(msg)

    run_dir = create_run_dir(model_group=input_group, cue=input_cue, target_value=input_target).resolve()
    save_used_config(config_path, run_dir)
    run_train(config, run_dir)


if __name__ == "__main__":
    main()
