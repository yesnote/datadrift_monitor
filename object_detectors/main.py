import argparse
import subprocess
import sys
from pathlib import Path

from dataloaders.dataloader_yolo import load_config
from commands.run_predict import run_grad_pass, run_predict_pass, should_run_grad_pass
from commands.utils.run_utils import create_run_dir, save_used_config

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent


def _resolve_path_value(raw_path):
    p = Path(raw_path)
    if p.is_absolute():
        return p.resolve()
    return (REPO_ROOT / p).resolve()


def _resolve_config_path(raw_path):
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def _resolve_run_dir(raw_path):
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def _normalize_config_paths(config):
    model_cfg = config.get("model", {})
    weight_path = model_cfg.get("weights")
    if isinstance(weight_path, str) and weight_path:
        model_cfg["weights"] = str(_resolve_path_value(weight_path))

    dataset_cfg = config.get("dataset", {})
    used_dataset = dataset_cfg.get("used_dataset")
    if isinstance(used_dataset, str) and used_dataset in dataset_cfg:
        active_cfg = dataset_cfg[used_dataset]
        root = active_cfg.get("root")
        if isinstance(root, str) and root:
            active_cfg["root"] = str(_resolve_path_value(root))

    return config


def run_subprocess_stage(config_path, stage, run_dir):
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config",
        str(config_path),
        "--stage",
        stage,
        "--run-dir",
        str(run_dir),
    ]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="object_detectors/configs/predict_yolov5.yaml")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "predict_pass", "grad_pass"],
    )
    parser.add_argument("--run-dir", type=str, default="")
    args = parser.parse_args()

    config_path = _resolve_config_path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = _normalize_config_paths(load_config(str(config_path)))
    mode = str(config.get("mode", "")).lower()
    if mode != "predict":
        raise ValueError(f"Unsupported mode: {mode}. Only 'predict' is implemented.")

    if args.stage == "all":
        run_dir = _resolve_run_dir(args.run_dir) if args.run_dir else create_run_dir().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        save_used_config(config_path, run_dir)

        run_subprocess_stage(config_path, "predict_pass", run_dir)
        if should_run_grad_pass(config):
            run_subprocess_stage(config_path, "grad_pass", run_dir)
        return

    if not args.run_dir:
        raise ValueError("--run-dir is required when --stage is 'predict_pass' or 'grad_pass'.")

    run_dir = _resolve_run_dir(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "predict_pass":
        run_predict_pass(config, run_dir)
        return

    run_grad_pass(config, run_dir)


if __name__ == "__main__":
    main()
