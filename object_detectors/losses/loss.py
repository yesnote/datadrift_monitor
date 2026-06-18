from losses.loss_yolo import build_yolo_loss
from models.yolov10.core import V10DetectLoss


def build_loss(model_type, model, config=None):
    model_type = model_type.lower()
    if model_type in {"yolo", "yolov5"}:
        return build_yolo_loss(model, config)
    if model_type in {"yolov10", "yolo_v10"}:
        return V10DetectLoss(model)
    raise ValueError(f"Unsupported model_type: {model_type}")
