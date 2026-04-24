import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import yaml

from commands.run_visualize import run_visualize

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent


def _resolve_config_path(raw_path: str) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p.resolve()
    return (REPO_ROOT / p).resolve()


def _resolve_run_dir(raw_path: str) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p.resolve()
    return (REPO_ROOT / p).resolve()


def _load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def _create_default_run_dir() -> Path:
    ts = datetime.now().strftime("%m-%d-%Y_%H;%M")
    out = PROJECT_ROOT / "runs" / f"{ts}_reference_pca"
    out.mkdir(parents=True, exist_ok=True)
    return out.resolve()


def _save_used_config(config_path: Path, run_dir: Path) -> None:
    shutil.copy2(config_path, run_dir / "used_config.yaml")


def _save_run_summary(run_dir: Path, summary: dict) -> None:
    out = run_dir / "run_summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="visualizers/configs/visualize_reference.yaml")
    parser.add_argument("--run-dir", type=str, default="")
    args = parser.parse_args()

    config_path = _resolve_config_path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    config = _load_config(config_path)
    mode = str(config.get("mode", "")).strip().lower()
    if mode != "visualize":
        raise ValueError("visualizers mode must be 'visualize'.")

    if args.run_dir:
        run_dir = _resolve_run_dir(args.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        cfg_out = str(config.get("output", {}).get("dir", "")).strip()
        if cfg_out:
            run_dir = _resolve_run_dir(cfg_out)
            run_dir.mkdir(parents=True, exist_ok=True)
        else:
            run_dir = _create_default_run_dir()

    _save_used_config(config_path, run_dir)
    summary = run_visualize(config, run_dir)
    _save_run_summary(run_dir, summary)
    print(f"Saved outputs to: {run_dir}")


if __name__ == "__main__":
    main()

