import shutil
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def create_run_dir(model_group: str, cue: str, target_value: str = "") -> Path:
    run_name = datetime.now().strftime("%m-%d-%Y_%H;%M")
    cue_name = str(cue).strip().lower()
    target_name = str(target_value).strip().lower()
    dir_name = f"{run_name}_{cue_name}" if not target_name else f"{run_name}_{cue_name}_{target_name}"
    group_name = str(model_group or "").strip().lower()
    safe_group = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in group_name).strip("_")
    base_dir = PROJECT_ROOT / "runs"
    if safe_group and safe_group != "bbox_predictions":
        base_dir = base_dir / safe_group
    run_dir = base_dir / dir_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_used_config(config_path: Path, run_dir: Path) -> None:
    shutil.copy2(config_path, run_dir / "used_config.yaml")
