from losses.loss_yolo import build_yolo_loss


def build_loss(model_type, model, config=None):
    model_type = model_type.lower()
    if model_type in {"yolo", "yolov5"}:
        return build_yolo_loss(model, config)
    raise ValueError(f"Unsupported model_type: {model_type}")
