import re
import sys
from contextlib import contextmanager
from pathlib import Path
from urllib.request import urlretrieve

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
from .core.utils import letterbox


YOLOV10_WEIGHT_URLS = {
    "n": "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.pt",
    "s": "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10s.pt",
    "m": "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10m.pt",
    "b": "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10b.pt",
    "l": "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10l.pt",
    "x": "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10x.pt",
}
DEFAULT_YOLOV10_ARCHITECTURE = "x"


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
        "architecture": payload.get("architecture"),
        "num_classes": payload.get("num_classes"),
        "names": payload.get("names"),
        "img_size": payload.get("img_size"),
    }


def _checkpoint_model_object(payload):
    if isinstance(payload, dict):
        model_payload = payload.get("ema")
        if model_payload is None:
            model_payload = payload.get("model")
        return model_payload if isinstance(model_payload, nn.Module) else None
    return payload if isinstance(payload, nn.Module) else None


def _normalize_yolov10_architecture(value):
    if value is None:
        return None
    text = str(value).lower().replace("\\", "/")
    match = re.search(r"yolov10([nsmblx])", text)
    if match:
        return match.group(1)
    text = text.replace("yolov10", "").strip()
    return text if text in YOLOV10_WEIGHT_URLS else None


def _infer_yolov10_architecture_from_payload(payload):
    metadata = _checkpoint_metadata(payload)
    inferred = _normalize_yolov10_architecture(metadata.get("architecture"))
    if inferred:
        return inferred
    model_payload = _checkpoint_model_object(payload)
    if model_payload is None:
        return None
    for attr in ("architecture", "yaml_file", "cfg", "model_name"):
        inferred = _normalize_yolov10_architecture(getattr(model_payload, attr, None))
        if inferred:
            return inferred
    yaml_obj = getattr(model_payload, "yaml", None)
    if isinstance(yaml_obj, dict):
        inferred = _normalize_yolov10_architecture(yaml_obj.get("yaml_file") or yaml_obj.get("model") or yaml_obj.get("name"))
        if inferred:
            return inferred
        scale = yaml_obj.get("scale")
        inferred = _normalize_yolov10_architecture(scale)
        if inferred:
            return inferred
        scales = yaml_obj.get("scales")
        if isinstance(scales, dict):
            candidates = [str(k).lower() for k in scales.keys() if str(k).lower() in YOLOV10_WEIGHT_URLS]
            if len(candidates) == 1:
                return candidates[0]
    return None


def _transform_state_dict_keys(state_dict, transform):
    return {transform(str(key)): value for key, value in state_dict.items()}


def _matching_state_dict_count(candidate, current):
    return sum(
        1
        for key, value in candidate.items()
        if key in current
        and hasattr(value, "shape")
        and tuple(current[key].shape) == tuple(value.shape)
    )


def _short_key_list(keys, limit=12):
    keys = list(keys)
    if len(keys) <= limit:
        return keys
    return keys[:limit] + [f"... (+{len(keys) - limit} more)"]


def _select_matching_state_dict(state_dict, current):
    candidates = [
        _transform_state_dict_keys(state_dict, lambda key: key),
        _transform_state_dict_keys(state_dict, lambda key: key[7:] if key.startswith("module.") else key),
    ]
    best = max(candidates, key=lambda candidate: _matching_state_dict_count(candidate, current))
    return best


def _infer_yolov10_architecture(path):
    return _normalize_yolov10_architecture(Path(path).name if path else None)


def _download_yolov10_weight(path):
    architecture = _infer_yolov10_architecture(path)
    url = YOLOV10_WEIGHT_URLS.get(architecture)
    if "coco" not in {part.lower() for part in path.parts}:
        raise FileNotFoundError(f"YOLOv10 weight file not found: {path}")
    if url is None:
        raise FileNotFoundError(f"YOLOv10 weight file not found and no official URL can be inferred from filename: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Downloading YOLOv10 {architecture} weight to {path}")
    urlretrieve(url, path)


class YOLOV10TorchObjectDetector(nn.Module):
    def __init__(
        self,
        model_weight=None,
        device="cuda",
        img_size=(640, 640),
        names=None,
        mode="eval",
        confidence=0.25,
        architecture=None,
        max_det=300,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.img_size = tuple(img_size)
        self.mode = str(mode)
        self.confidence = float(confidence)
        self.max_det = int(max_det)
        self.names = list(names or [])
        self.num_classes = len(self.names) if self.names else 80
        preloaded_payload = None
        inferred = architecture or _infer_yolov10_architecture(model_weight)
        if model_weight and Path(model_weight).is_file() and not inferred:
            preloaded_payload = _torch_load(Path(model_weight), self.device)
            inferred = _infer_yolov10_architecture_from_payload(preloaded_payload)
        self.architecture = str(inferred or DEFAULT_YOLOV10_ARCHITECTURE).lower().replace("yolov10", "")
        if not self.architecture:
            raise ValueError(
                "YOLOv10 architecture must be inferable from model.weights filename or checkpoint metadata/model yaml."
            )
        self.is_yolov10 = True
        self.agnostic = False
        cfg = load_yolov10_cfg(self.architecture)
        self.model = YOLOv10DetectionModel(cfg, nc=self.num_classes)
        self.model.names = self.names or [str(i) for i in range(self.num_classes)]
        self.model.architecture = self.architecture
        self.names = self.model.names
        if model_weight:
            self.load_weights(model_weight, payload=preloaded_payload)
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

    @staticmethod
    def yolo_resize(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
        return letterbox(img, new_shape=new_shape, color=color, auto=auto, scaleFill=scaleFill, scaleup=scaleup)

    def load_weights(self, model_weight, payload=None):
        path = Path(model_weight)
        if not path.is_file():
            _download_yolov10_weight(path)
        payload = payload if payload is not None else _torch_load(path, self.device)
        metadata = _checkpoint_metadata(payload)
        if metadata.get("model_type") and str(metadata["model_type"]).lower() != "yolov10":
            raise ValueError(f"Checkpoint model_type is not yolov10: {metadata['model_type']}")
        payload_architecture = _infer_yolov10_architecture_from_payload(payload)
        if payload_architecture and payload_architecture != self.architecture:
            raise ValueError(f"YOLOv10 checkpoint architecture mismatch: checkpoint={payload_architecture}, model={self.architecture}")
        if metadata.get("num_classes") is not None and int(metadata["num_classes"]) != int(self.num_classes):
            print(f"[WARN] YOLOv10 checkpoint num_classes mismatch: checkpoint={metadata['num_classes']}, config={self.num_classes}")
        current = self.model.state_dict()
        state_dict = _select_matching_state_dict(_checkpoint_state_dict(payload), current)
        current_keys = set(current.keys())
        state_keys = set(state_dict.keys())
        matching_keys = {
            key
            for key in current_keys & state_keys
            if tuple(current[key].shape) == tuple(state_dict[key].shape)
        }
        if not matching_keys:
            raise ValueError(f"YOLOv10 checkpoint has no matching parameters: {path}")
        missing_keys = sorted(current_keys - matching_keys)
        unexpected_keys = sorted(state_keys - current_keys)
        shape_mismatch_keys = sorted(
            key
            for key in current_keys & state_keys
            if tuple(current[key].shape) != tuple(state_dict[key].shape)
        )
        if missing_keys or unexpected_keys or shape_mismatch_keys:
            raise ValueError(
                "YOLOv10 checkpoint is not fully compatible with the local model. "
                f"loaded={len(matching_keys)}, model_keys={len(current_keys)}, checkpoint_keys={len(state_keys)}, "
                f"missing={_short_key_list(missing_keys)}, "
                f"unexpected={_short_key_list(unexpected_keys)}, "
                f"shape_mismatch={_short_key_list(shape_mismatch_keys)}"
            )
        self.model.load_state_dict(state_dict, strict=True)

    def build_loss(self):
        return V10DetectLoss(self.model)

    def _input_shape_hw(self, images=None, input_shape=None):
        if input_shape is not None:
            return int(input_shape[-2]), int(input_shape[-1])
        if images is not None:
            return int(images.shape[-2]), int(images.shape[-1])
        return int(self.img_size[0]), int(self.img_size[1])

    @staticmethod
    def _clip_xyxy_to_shape(boxes, input_shape):
        if boxes.numel() == 0:
            return boxes
        h, w = int(input_shape[0]), int(input_shape[1])
        clipped = boxes.clone()
        clipped[:, [0, 2]] = clipped[:, [0, 2]].clamp(0, max(0, w - 1))
        clipped[:, [1, 3]] = clipped[:, [1, 3]].clamp(0, max(0, h - 1))
        return clipped

    @staticmethod
    def _xyxy_validity_score(boxes, input_shape, source_points=None):
        if boxes.numel() == 0:
            return 1.0
        h, w = float(input_shape[0]), float(input_shape[1])
        x1, y1, x2, y2 = boxes.unbind(-1)
        bw = x2 - x1
        bh = y2 - y1
        finite = torch.isfinite(boxes).all(dim=-1)
        ordered = (bw >= 0) & (bh >= 0)
        margin_w = max(w, 1.0) * 0.5
        margin_h = max(h, 1.0) * 0.5
        not_far = (x2 >= -margin_w) & (y2 >= -margin_h) & (x1 <= w + margin_w) & (y1 <= h + margin_h)
        not_huge = (bw <= max(w, 1.0) * 2.5) & (bh <= max(h, 1.0) * 2.5)
        valid = finite & ordered & not_far & not_huge
        if source_points is not None and source_points.numel() > 0:
            px, py = source_points.unbind(-1)
            contains_source = (px >= x1) & (px <= x2) & (py >= y1) & (py <= y2)
            valid = valid & contains_source
        return float(valid.float().mean().detach().cpu().item())

    def _canonical_xyxy_from_postprocess_boxes(self, boxes, input_shape, context, source_points=None):
        xywh_as_xyxy = xywh2xyxy(boxes)
        xyxy_as_xyxy = boxes.clone()
        xywh_score = self._xyxy_validity_score(xywh_as_xyxy, input_shape, source_points=source_points)
        xyxy_score = self._xyxy_validity_score(xyxy_as_xyxy, input_shape, source_points=source_points)
        h, w = float(input_shape[0]), float(input_shape[1])
        scale = boxes.new_tensor([max(w, 1.0), max(h, 1.0), max(w, 1.0), max(h, 1.0)])
        format_delta = 0.0
        if boxes.numel() > 0:
            format_delta = float(((xywh_as_xyxy - xyxy_as_xyxy).abs() / scale).mean().detach().cpu().item())
        if xywh_score >= 0.75 and xywh_score >= xyxy_score + 0.25:
            chosen = xywh_as_xyxy
        elif xyxy_score >= 0.75 and xyxy_score >= xywh_score + 0.25:
            chosen = xyxy_as_xyxy
            if not getattr(self, "_warned_yolov10_xyxy_postprocess", False):
                print(
                    "[WARN] YOLOv10 postprocess boxes look like xyxy, not xywh; "
                    "using xyxy interpretation for selected predictions."
                )
                self._warned_yolov10_xyxy_postprocess = True
        elif xywh_score >= 0.75 and xyxy_score >= 0.75 and format_delta > 0.2:
            raise RuntimeError(
                "YOLOv10 decoded box format is ambiguous; refusing to save potentially mis-scaled boxes. "
                f"context={context}, input_shape={tuple(int(v) for v in input_shape)}, "
                f"xywh_score={xywh_score:.3f}, xyxy_score={xyxy_score:.3f}, format_delta={format_delta:.3f}"
            )
        elif xywh_score >= 0.75:
            chosen = xywh_as_xyxy
        elif xyxy_score >= 0.75:
            chosen = xyxy_as_xyxy
            if not getattr(self, "_warned_yolov10_xyxy_postprocess", False):
                print(
                    "[WARN] YOLOv10 postprocess boxes look like xyxy, not xywh; "
                    "using xyxy interpretation for selected predictions."
                )
                self._warned_yolov10_xyxy_postprocess = True
        else:
            raise RuntimeError(
                "YOLOv10 decoded boxes are not plausible in either xywh or xyxy interpretation. "
                f"context={context}, input_shape={tuple(int(v) for v in input_shape)}, "
                f"xywh_score={xywh_score:.3f}, xyxy_score={xyxy_score:.3f}, format_delta={format_delta:.3f}"
            )
        return self._clip_xyxy_to_shape(chosen, input_shape)

    def _decode_eval_output(self, model_output, input_shape=None):
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
        selected_indices = []
        source_points = self.source_points(raw_levels)
        for sample_idx in range(decoded_bnc.shape[0]):
            keep = scores[sample_idx] >= self.confidence
            b = boxes_xywh[sample_idx][keep]
            s = scores[sample_idx][keep]
            l = labels[sample_idx][keep].float()
            idx = raw_indices[sample_idx][keep].long()
            raw_box_idx = torch.div(idx, self.num_classes, rounding_mode="floor")
            selected_source_points = source_points[raw_box_idx]
            xyxy = self._canonical_xyxy_from_postprocess_boxes(
                b,
                input_shape,
                context=f"sample_idx={sample_idx}",
                source_points=selected_source_points,
            )
            selected_preds.append(torch.cat([xyxy, s[:, None], l[:, None]], dim=1))
            selected_indices.append(idx)
        return selected_preds, selected_indices, raw_levels, decoded_bnc

    def prepare_feature_cache(self, images):
        return self.model.forward_features(images)

    def forward_raw(self, images=None, feature_cache=None):
        if feature_cache is not None:
            return self.model.forward_one2one_from_features(feature_cache)
        return self.model.forward_one2one(images)

    def forward_raw_decoded(self, images=None, feature_cache=None, source_points=None):
        model_output = self.forward_raw(images, feature_cache=feature_cache)
        one2one = model_output["one2one"]
        if not isinstance(one2one, (tuple, list)) or len(one2one) < 2:
            raise RuntimeError("YOLOv10 one2one inference output must be (decoded, raw_levels).")
        decoded, raw_levels = one2one
        decoded_bnc = decoded.permute(0, 2, 1).contiguous()
        return {
            "model_output": model_output,
            "raw_levels": raw_levels,
            "decoded_prediction": decoded_bnc,
            "raw_logits": self.flatten_class_logits(raw_levels),
            "source_points": source_points if source_points is not None else self.source_points(raw_levels),
        }

    def forward_nms_free(self, images=None, feature_cache=None, source_points=None, input_shape=None):
        raw_output = self.forward_raw_decoded(images, feature_cache=feature_cache, source_points=source_points)
        selected_preds, selected_indices, raw_levels, decoded_bnc = self._decode_eval_output(
            raw_output["model_output"],
            input_shape=self._input_shape_hw(images=images, input_shape=input_shape),
        )
        return {
            "model_output": raw_output["model_output"],
            "raw_levels": raw_levels,
            "decoded_prediction": decoded_bnc,
            "raw_logits": raw_output["raw_logits"],
            "selected_preds": selected_preds,
            "selected_indices": selected_indices,
            "source_points": raw_output["source_points"],
        }

    def forward_layer_grad(self, images, source_points=None):
        was_training = self.model.training
        self.model.eval()
        model_output = self.forward_raw(images)
        selected_preds, selected_indices, raw_levels, decoded_bnc = self._decode_eval_output(
            model_output,
            input_shape=self._input_shape_hw(images=images),
        )
        if was_training and self.mode == "train":
            self.model.train()
        return {
            "model_output": model_output,
            "raw_levels": raw_levels,
            "decoded_prediction": decoded_bnc,
            "raw_logits": self.flatten_class_logits(raw_levels),
            "selected_preds": selected_preds,
            "selected_indices": selected_indices,
            "source_points": source_points if source_points is not None else self.source_points(raw_levels),
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
        return [boxes, classes, class_names, confidences], output["raw_logits"], None, None


__all__ = ["YOLOV10TorchObjectDetector"]
