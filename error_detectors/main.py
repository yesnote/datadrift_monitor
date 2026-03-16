import argparse
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


def parse_root_info(root_path: Path) -> tuple[str, str]:
    # New format: .../runs/{model_group}/{cue}/{time}
    # Backward-compatible old format: .../runs/{model_group}/{time}_{cue}
    parent = root_path.parent
    if parent.name in {"fn_detectors", "tp_classifiers"}:
        model_group = parent.name
        run_name = root_path.name
        cue = run_name.split("_", 2)[-1] if "_" in run_name else run_name
        return model_group, cue

    if parent.parent.name in {"fn_detectors", "tp_classifiers"}:
        model_group = parent.parent.name
        cue = parent.name
        return model_group, cue

    raise ValueError(
        "dataset root must follow object_detectors/runs/{fn_detectors|tp_classifiers}/{cue}/{time} "
        "or legacy object_detectors/runs/{fn_detectors|tp_classifiers}/{time}_{cue}."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="error_detectors/configs/train_error_detector.yaml")
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = load_config(config_path)
    mode = str(config.get("mode", "train")).lower()
    if mode != "train":
        raise ValueError(f"Unsupported mode: {mode}. Only 'train' is supported.")

    dataset_cfg = config.get("dataset", {})
    input_root_raw = str(dataset_cfg.get("input_root", "")).strip()
    gt_root_raw = str(dataset_cfg.get("gt_root", "")).strip()
    if not input_root_raw or not gt_root_raw:
        raise ValueError("dataset.input_root and dataset.gt_root are required.")

    input_root = resolve_path_value(input_root_raw)
    gt_root = resolve_path_value(gt_root_raw)
    input_group, input_cue = parse_root_info(input_root)
    gt_group, _gt_cue = parse_root_info(gt_root)
    if input_group != gt_group:
        msg = (
            "dataset.input_root and dataset.gt_root must have the same model group "
            f"(got '{input_group}' vs '{gt_group}')."
        )
        warnings.warn(msg)
        raise ValueError(msg)

    run_dir = create_run_dir(model_group=input_group, cue=input_cue).resolve()
    save_used_config(config_path, run_dir)
    run_train(config, run_dir)


if __name__ == "__main__":
    main()
