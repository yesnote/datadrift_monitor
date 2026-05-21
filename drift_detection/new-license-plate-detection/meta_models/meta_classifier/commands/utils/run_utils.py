import shutil
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _safe_path_part(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value.strip().lower()).strip("_")


def create_run_dir(model_group: str, cue: str, target_value: str = "", mode_subdir: str = "") -> Path:
    run_name = datetime.now().strftime("%m-%d-%Y_%H;%M")
    cue_name = str(cue).strip().lower()
    target_name = str(target_value).strip().lower()
    dir_name = f"{run_name}_{cue_name}" if not target_name else f"{run_name}_{cue_name}_{target_name}"
    raw_parts = [
        part
        for part in str(model_group or "").replace("\\", "/").split("/")
        if _safe_path_part(part) and _safe_path_part(part) != "bbox_predictions"
    ]
    group_parts = [_safe_path_part(part) for part in raw_parts]
    safe_mode = _safe_path_part(str(mode_subdir or ""))

    base_dir = PROJECT_ROOT / "runs"

    if len(group_parts) >= 2:
        # New layout: runs/<object_detector_model>/<mode>/<dataset>/<optional_group...>/<time>...
        base_dir = base_dir / group_parts[0]
        if safe_mode:
            base_dir = base_dir / safe_mode
        for part in group_parts[1:]:
            base_dir = base_dir / part
    else:
        # Legacy fallback for old object_detectors/runs/<dataset>/<time> inputs.
        if safe_mode:
            base_dir = base_dir / safe_mode
        for part in group_parts:
            base_dir = base_dir / part

    run_dir = base_dir / dir_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_used_config(config_path: Path, run_dir: Path) -> None:
    shutil.copy2(config_path, run_dir / "used_config.yaml")
