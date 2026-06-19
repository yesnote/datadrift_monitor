import sys
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn

from .core import (
    V10DetectLoss,
    YOLOv10DetectionModel,
    load_yolov10_cfg,
    make_anchors,
    v10postprocess_with_indices,
    xywh2xyxy,
)


def _torch_load(path, map_location):
    ref_root = Path(__file__).resolve().parents[3] / "code_references" / "yolov10-main"
    with _temporary_sys_path(ref_root if ref_root.is_dir() else None):
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=map_location)


@contextmanager
def _temporary_sys_path(path):
    if path is None:
        yield
        return
    token = str(path)
    inserted = token not in sys.path
    if inserted:
        sys.path.insert(0, token)
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(token)
            except ValueError:
                pass


def _checkpoint_state_dict(payload):
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    if isinstance(payload, dict) and payload.get("ema") is not None:
        model_payload = payload["ema"]
        if isinstance(model_payload, nn.Module):
            return model_payload.state_dict()
        if isinstance(model_payload, dict):
            return model_payload
    if isinstance(payload, dict) and "model" in payload:
        model_payload = payload["model"]
        if isinstance(model_payload, nn.Module):
            return model_payload.state_dict()
        if isinstance(model_payload, dict):
            return model_payload
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"]
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, nn.Module):
        return payload.state_dict()
    raise ValueError("Unsupported YOLOv10 checkpoint payload.")


def _checkpoint_metadata(payload):
    if not isinstance(payload, dict):
        return {}
    return {
        "model_type": payload.get("model_type"),
        "variant": payload.get("variant"),
        "num_classes": payload.get("num_classes"),
        "names": payload.get("names"),
        "img_size": payload.get("img_size"),
    }


def _strip_prefixes(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        new_key = str(key)
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "model."):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True
        cleaned[new_key] = value
    return cleaned


class YOLOV10TorchObjectDetector(nn.Module):
    def __init__(
        self,
        model_weight=None,
        device="cuda",
        img_size=(640, 640),
        names=None,
        mode="eval",
        confidence=0.25,
        iou_thresh=0.45,
        variant="n",
        max_det=300,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.img_size = tuple(img_size)
        self.mode = str(mode)
        self.confidence = float(confidence)
        self.iou_thresh = float(iou_thresh)
        self.max_det = int(max_det)
        self.variant = str(variant or "n").lower().replace("yolov10", "")
        self.is_yolov10 = True
        self.agnostic = False
        self.names = list(names or [])
        self.num_classes = len(self.names) if self.names else 80
        cfg = load_yolov10_cfg(self.variant)
        self.model = YOLOv10DetectionModel(cfg, nc=self.num_classes)
        self.model.names = self.names or [str(i) for i in range(self.num_classes)]
        self.names = self.model.names
        if model_weight:
            self.load_weights(model_weight)
        self.model.to(self.device)
        if self.mode == "train":
            self.model.train()
        else:
            self.model.eval()
        print("[INFO] YOLOv10 model is loaded")

    def train(self, mode=True):
        super().train(mode)
        self.model.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def load_weights(self, model_weight):
        path = Path(model_weight)
        if not path.is_file():
            raise FileNotFoundError(f"YOLOv10 weight file not found: {path}")
        payload = _torch_load(path, self.device)
        metadata = _checkpoint_metadata(payload)
        if metadata.get("model_type") and str(metadata["model_type"]).lower() != "yolov10":
            raise ValueError(f"Checkpoint model_type is not yolov10: {metadata['model_type']}")
        if metadata.get("variant") and str(metadata["variant"]).lower().replace("yolov10", "") != self.variant:
            raise ValueError(f"YOLOv10 checkpoint variant mismatch: checkpoint={metadata['variant']}, config={self.variant}")
        if metadata.get("num_classes") is not None and int(metadata["num_classes"]) != int(self.num_classes):
            raise ValueError(
                f"YOLOv10 checkpoint num_classes mismatch: checkpoint={metadata['num_classes']}, config={self.num_classes}"
            )
        state_dict = _strip_prefixes(_checkpoint_state_dict(payload))
        current = self.model.state_dict()
        filtered = {
            key: value
            for key, value in state_dict.items()
            if key in current and tuple(current[key].shape) == tuple(value.shape)
        }
        skipped = len(state_dict) - len(filtered)
        if state_dict and len(filtered) / max(1, len(current)) < 0.5:
            raise ValueError(
                f"YOLOv10 checkpoint load ratio is too low: matched={len(filtered)}, model_keys={len(current)}, "
                f"checkpoint_keys={len(state_dict)}"
            )
        if skipped:
            print(f"[WARN] YOLOv10 checkpoint partial load: loaded={len(filtered)}, skipped={skipped}")
        self.model.load_state_dict(filtered, strict=False)

    def build_loss(self):
        return V10DetectLoss(self.model)

    def _decode_eval_output(self, model_output):
        one2one = model_output["one2one"]
        if not isinstance(one2one, (tuple, list)) or len(one2one) < 2:
            raise RuntimeError("YOLOv10 one2one inference output must be (decoded, raw_levels).")
        decoded, raw_levels = one2one
        decoded_bnc = decoded.permute(0, 2, 1).contiguous()
        boxes_xywh, scores, labels, raw_indices = v10postprocess_with_indices(
            decoded_bnc,
            max_det=self.max_det,
            nc=self.num_classes,
        )
        selected_preds = []
        selected_logits = []
        selected_probs = []
        selected_indices = []
        flat_logits = self.flatten_class_logits(raw_levels)
        for sample_idx in range(decoded_bnc.shape[0]):
            keep = scores[sample_idx] >= self.confidence
            b = boxes_xywh[sample_idx][keep]
            s = scores[sample_idx][keep]
            l = labels[sample_idx][keep].float()
            idx = raw_indices[sample_idx][keep].long()
            raw_box_idx = torch.div(idx, self.num_classes, rounding_mode="floor")
            xyxy = xywh2xyxy(b)
            selected_preds.append(torch.cat([xyxy, s[:, None], l[:, None]], dim=1))
            selected_indices.append(idx)
            selected_logits.append(flat_logits[sample_idx][raw_box_idx].detach())
            selected_probs.append(torch.sigmoid(flat_logits[sample_idx][raw_box_idx]).detach())
        return selected_preds, selected_logits, selected_probs, selected_indices, raw_levels, decoded_bnc

    def forward_raw(self, images):
        return self.model(images, augment=False)

    def forward_nms_free(self, images):
        model_output = self.forward_raw(images)
        selected_preds, selected_logits, selected_probs, selected_indices, raw_levels, decoded_bnc = self._decode_eval_output(model_output)
        return {
            "model_output": model_output,
            "raw_levels": raw_levels,
            "decoded_prediction": decoded_bnc,
            "raw_logits": self.flatten_class_logits(raw_levels),
            "selected_preds": selected_preds,
            "selected_logits": selected_logits,
            "selected_probs": selected_probs,
            "selected_indices": selected_indices,
            "source_points": self.source_points(raw_levels),
        }

    def forward_layer_grad(self, images):
        was_training = self.model.training
        self.model.eval()
        model_output = self.model(images, augment=False)
        selected_preds, selected_logits, selected_probs, selected_indices, raw_levels, decoded_bnc = self._decode_eval_output(model_output)
        if was_training and self.mode == "train":
            self.model.train()
        return {
            "model_output": model_output,
            "raw_levels": raw_levels,
            "decoded_prediction": decoded_bnc,
            "raw_logits": self.flatten_class_logits(raw_levels),
            "selected_preds": selected_preds,
            "selected_logits": selected_logits,
            "selected_probs": selected_probs,
            "selected_indices": selected_indices,
            "source_points": self.source_points(raw_levels),
        }

    def flatten_raw_levels(self, raw_levels):
        head = self.model.model[-1]
        return torch.cat([x.view(x.shape[0], head.no, -1) for x in raw_levels], dim=2).permute(0, 2, 1).contiguous()

    def flatten_class_logits(self, raw_levels):
        head = self.model.model[-1]
        raw = self.flatten_raw_levels(raw_levels)
        return raw[..., head.reg_max * 4 :]

    def flatten_distri(self, raw_levels):
        head = self.model.model[-1]
        raw = self.flatten_raw_levels(raw_levels)
        return raw[..., : head.reg_max * 4]

    def source_points(self, raw_levels):
        head = self.model.model[-1]
        anchor_points, stride_tensor = make_anchors(raw_levels, head.stride, 0.5)
        return anchor_points * stride_tensor

    def forward(self, images, *args, **kwargs):
        output = self.forward_nms_free(images)
        boxes, classes, class_names, confidences = [], [], [], []
        for det in output["selected_preds"]:
            boxes_b, classes_b, names_b, conf_b = [], [], [], []
            for row in det:
                cls_idx = int(row[5].detach().cpu().item())
                boxes_b.append([float(v) for v in row[:4].detach().cpu().tolist()])
                classes_b.append(cls_idx)
                names_b.append(self.names[cls_idx] if 0 <= cls_idx < len(self.names) else str(cls_idx))
                conf_b.append(float(row[4].detach().cpu().item()))
            boxes.append(boxes_b)
            classes.append(classes_b)
            class_names.append(names_b)
            confidences.append(conf_b)
        return [boxes, classes, class_names, confidences], output["selected_logits"], None, None


__all__ = ["YOLOV10TorchObjectDetector"]
