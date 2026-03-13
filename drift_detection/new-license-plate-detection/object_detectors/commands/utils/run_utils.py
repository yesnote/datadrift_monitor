import shutil
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def create_run_dir(cue=None, unit=None):
    run_name = datetime.now().strftime("%m-%d-%Y_%H;%M")
    cue_name = str(cue or "predict").lower()
    unit_name = str(unit or "").lower()

    if unit_name == "image":
        base_dir = PROJECT_ROOT / "runs" / "fn_detectors"
    elif unit_name == "bbox":
        base_dir = PROJECT_ROOT / "runs" / "tp_classifiers"
    else:
        base_dir = PROJECT_ROOT / "runs"

    run_dir = base_dir / f"{run_name}_{cue_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_used_config(config_path, run_dir):
    shutil.copy2(config_path, run_dir / "used_config.yaml")
