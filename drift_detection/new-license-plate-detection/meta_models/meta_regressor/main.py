import argparse
import sys
import warnings
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from commands.run_train import run_train
from commands.run_test import run_test
from commands.utils.run_utils import create_run_dir, save_used_config
from meta_models.common import normalize_input_roots, parse_root_info, resolve_path_value


def resolve_config_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="meta_models/meta_regressor/configs/train_meta_regressor.yaml")
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = load_config(config_path)

    mode = str(config.get("mode", "train")).strip().lower()
    if mode not in {"train", "test"}:
        raise ValueError(f"Unsupported mode: {mode}. Use 'train' or 'test'.")

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

    cue_for_run = input_cue if mode == "train" else f"{input_cue}_meta_regressor_test"
    mode_subdir = mode
    run_dir = create_run_dir(
        model_group=input_group,
        cue=cue_for_run,
        target_value=input_target,
        mode_subdir=mode_subdir,
    ).resolve()
    save_used_config(config_path, run_dir)
    if mode == "train":
        run_train(config, run_dir)
    else:
        run_test(config, run_dir)


if __name__ == "__main__":
    main()
