import sys
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent
OD_ROOT = REPO_ROOT / "object_detectors"
if str(OD_ROOT) not in sys.path:
    sys.path.insert(0, str(OD_ROOT))

from models.yolo.models.yolo_v5_object_detector import YOLOV5TorchObjectDetector


def main() -> None:
    config_path = Path("object_detectors/configs/predict_yolov5.yaml")
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    device_str = model_cfg.get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    weight_path = Path(cfg["model"]["weights"])
    if not weight_path.is_absolute():
        weight_path = (REPO_ROOT / weight_path).resolve()

    detector = YOLOV5TorchObjectDetector(
        model_weight=str(weight_path),
        device=device,
        img_size=(model_cfg["img_size"], model_cfg["img_size"]),
        names=None,
        mode="eval",
        confidence=model_cfg.get("confidence_threshold", 0.4),
        iou_thresh=model_cfg.get("iou_threshold", 0.45),
    )
    detector.model.requires_grad_(False)
    detector.eval().to(device)

    module_names = [name for name, _module in detector.model.named_modules() if name]
    if len(module_names) < 2:
        print("Not enough module names found.")
        return

    print("Last 200 module names:")
    for name in module_names[-200:]:
        print(name)


if __name__ == "__main__":
    main()
