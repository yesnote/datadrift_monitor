from types import MethodType

import torch
import torch.nn.functional as F

from models.yolov10.core import xywh2xyxy


def yolov10_raw_logits_for_item(forward, item, device):
    return forward.raw_logits[item["sample_idx"], item["raw_box_idx"]].to(device=device, dtype=torch.float32)


def yolov10_raw_probs_for_item(forward, item, device):
    return torch.sigmoid(yolov10_raw_logits_for_item(forward, item, device))


def source_point_box(source_points, raw_box_idx, device):
    point = source_points[int(raw_box_idx)].to(device=device, dtype=torch.float32)
    return torch.stack([point[0], point[1], point[0], point[1]])


def gather_yolov10_feature_matrix(forward, items, device):
    num_classes = int(forward.raw_logits.shape[-1])
    if not items:
        return torch.zeros((0, 4 + num_classes), dtype=torch.float32, device=device)
    source_device = forward.decoded_prediction.device
    sample_idx = torch.as_tensor([item["sample_idx"] for item in items], dtype=torch.long, device=source_device)
    raw_box_idx = torch.as_tensor([item["raw_box_idx"] for item in items], dtype=torch.long, device=source_device)
    boxes_xywh = forward.decoded_prediction[sample_idx, raw_box_idx, :4].detach().float()
    boxes_xyxy = xywh2xyxy(boxes_xywh)
    probs = torch.sigmoid(forward.raw_logits[sample_idx, raw_box_idx].detach().float())
    return torch.cat([boxes_xyxy.to(device=device), probs.to(device=device)], dim=1)


def enable_forced_yolov10_dropout(model, dropout_rate):
    head = model.model[-1]
    original = head.forward_feat
    p = float(dropout_rate)

    def patched(self, x, cv2, cv3):
        outputs = original(x, cv2, cv3)
        return [F.dropout(out, p=p, training=True) for out in outputs]

    head.forward_feat = MethodType(patched, head)

    def restore():
        head.forward_feat = original

    return restore


__all__ = [
    "enable_forced_yolov10_dropout",
    "gather_yolov10_feature_matrix",
    "source_point_box",
    "yolov10_raw_logits_for_item",
    "yolov10_raw_probs_for_item",
]
