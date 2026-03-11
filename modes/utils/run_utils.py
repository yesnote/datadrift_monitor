import shutil
from datetime import datetime
from pathlib import Path


def create_run_dir():
    run_name = datetime.now().strftime("%m-%d-%Y_%H;%M")
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_used_config(config_path, run_dir):
    shutil.copy2(config_path, run_dir / "used_config.yaml")
