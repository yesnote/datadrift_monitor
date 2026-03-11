from models.yolo.utils.loss import ComputeLoss


class YoloLoss:
    def __init__(self, model):
        self.compute_loss = ComputeLoss(model)

    def __call__(self, predictions, targets):
        return self.compute_loss(predictions, targets)


def build_yolo_loss(model, _config=None):
    return YoloLoss(model)
