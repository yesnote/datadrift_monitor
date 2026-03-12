import argparse
import subprocess
import sys
from pathlib import Path

from dataloaders.dataloader_yolo import load_config
from modes.run_predict import run_grad_pass, run_predict_pass, should_run_grad_pass
from modes.utils.run_utils import create_run_dir, save_used_config


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
    parser.add_argument("--config", type=str, default="configs/predict_yolov5.yaml")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "predict_pass", "grad_pass"],
    )
    parser.add_argument("--run-dir", type=str, default="")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = load_config(str(config_path))
    mode = str(config.get("mode", "")).lower()
    if mode != "predict":
        raise ValueError(f"Unsupported mode: {mode}. Only 'predict' is implemented.")

    if args.stage == "all":
        run_dir = Path(args.run_dir).resolve() if args.run_dir else create_run_dir().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        save_used_config(config_path, run_dir)

        run_subprocess_stage(config_path, "predict_pass", run_dir)
        if should_run_grad_pass(config):
            run_subprocess_stage(config_path, "grad_pass", run_dir)
        return

    if not args.run_dir:
        raise ValueError("--run-dir is required when --stage is 'predict_pass' or 'grad_pass'.")

    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "predict_pass":
        run_predict_pass(config, run_dir)
        return

    run_grad_pass(config, run_dir)


if __name__ == "__main__":
    main()
