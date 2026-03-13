import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fn_detectors.commands.run_train import run_train
from fn_detectors.commands.utils.run_utils import create_run_dir, save_used_config


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
    parser.add_argument("--config", type=str, default="fn_detectors/configs/train_fn_detector.yaml")
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = load_config(config_path)
    mode = str(config.get("mode", "train")).lower()
    if mode != "train":
        raise ValueError(f"Unsupported mode: {mode}. Only 'train' is supported.")

    run_dir = create_run_dir().resolve()
    save_used_config(config_path, run_dir)
    run_train(config, run_dir)


if __name__ == "__main__":
    main()
