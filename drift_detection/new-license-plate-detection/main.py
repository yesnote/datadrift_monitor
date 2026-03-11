import argparse
from pathlib import Path

from dataloaders.dataloader_yolo import load_config
from modes.run_predict import run_predict
from modes.utils.run_utils import create_run_dir, save_used_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/predict_yolov5.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = load_config(str(config_path))
    mode = str(config.get("mode", "")).lower()

    run_dir = create_run_dir()
    save_used_config(config_path, run_dir)

    if mode == "predict":
        run_predict(config, run_dir)
        return

    raise ValueError(f"Unsupported mode: {mode}. Only 'predict' is implemented.")


if __name__ == "__main__":
    main()
