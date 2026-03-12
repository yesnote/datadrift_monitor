import argparse
from pathlib import Path

import yaml

from fn_detectors.commands.run_train import run_train


PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_config_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.is_file():
        return cwd_candidate
    return (PROJECT_ROOT / path).resolve()


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_fn_detector.yaml")
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = load_config(config_path)
    mode = str(config.get("mode", "train")).lower()
    if mode != "train":
        raise ValueError(f"Unsupported mode: {mode}. Only 'train' is supported.")

    run_train(config)


if __name__ == "__main__":
    main()

