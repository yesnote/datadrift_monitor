import shutil
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def create_run_dir(model_group: str, cue: str) -> Path:
    run_name = datetime.now().strftime("%m-%d-%Y_%H;%M")
    group = str(model_group).strip().lower()
    cue_name = str(cue).strip().lower()
    run_dir = PROJECT_ROOT / "runs" / group / cue_name / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_used_config(config_path: Path, run_dir: Path) -> None:
    shutil.copy2(config_path, run_dir / "used_config.yaml")
