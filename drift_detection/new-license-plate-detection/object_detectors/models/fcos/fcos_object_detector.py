from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as TF


FCOS_PACKAGE_ROOT = Path(__file__).resolve().parent
if str(FCOS_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(FCOS_PACKAGE_ROOT))

DEFAULT_FCOS_COCO_WEIGHT = FCOS_PACKAGE_ROOT / "weights" / "coco" / "FCOS_R_50_FPN_1x.pth"
DEFAULT_FCOS_COCO_WEIGHT_URL = (
    "https://huggingface.co/tianzhi/FCOS/resolve/main/FCOS_R_50_FPN_1x.pth?download=true"
)


def _torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _checkpoint_state_dict(payload):
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
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
    raise ValueError("Unsupported FCOS checkpoint payload.")


def _strip_prefixes(state_dict):
    prefixes = ("module.", "model.")
    cleaned = {}
    for key, value in state_dict.items():
        new_key = str(key)
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
        cleaned[new_key] = value
    return cleaned


def _ensure_fcos_weight(path):
    path = Path(path)
    if path.is_file():
        return path
    if path.name != DEFAULT_FCOS_COCO_WEIGHT.name:
        raise FileNotFoundError(f"FCOS weight file not found: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Downloading FCOS COCO weight to {path}")
    torch.hub.download_url_to_file(DEFAULT_FCOS_COCO_WEIGHT_URL, str(path), progress=True)
    return path


class _ForwardProxy:
    def __init__(self, owner):
        self._owner = owner

    def __call__(self, images, *args, **kwargs):
        return self._owner.forward(images)

    def __getattr__(self, name):
        return getattr(self._owner.detector_model, name)


class FCOSTorchObjectDetector(nn.Module):
    def __init__(
        self,
        model_weight=None,
        device="cuda",
        names=None,
        mode="eval",
        confidence=0.05,
        iou_thresh=0.6,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.mode = str(mode)
        self.confidence = float(confidence)
        self.conf_thresh = float(confidence)
        self.iou_thresh = float(iou_thresh)
        self.agnostic = False
        self.agnostic_nms = False
        self.is_fcos = True
        self.has_faster_rcnn_label_column = True
        self._dropout_handles = []
        self._dropout_rate = 0.0

        self.names = list(names or [])
        self.num_classes_no_bg = len(self.names) if self.names else 80

        self.cfg = self._build_cfg()
        self.max_det = int(self.cfg.TEST.DETECTIONS_PER_IMG)
        self.detector_model = self._build_model()
        self.model = _ForwardProxy(self)
        self._resize_transform, self._normalize_transform = self._build_preprocess_transforms()

        if model_weight:
            self.load_weights(model_weight)
        self.to(self.device)
        if self.mode == "train":
            self.train()
        else:
            self.eval()

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, "detector_model"):
            self.detector_model.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def _build_cfg(self):
        try:
            from fcos_core.config import cfg as base_cfg
        except ImportError as exc:
            raise ImportError(
                "FCOS config requires yacs and the copied fcos_core package. "
                "Install the FCOS reference dependencies if this import fails."
            ) from exc

        cfg = base_cfg.clone()

        cfg.defrost()
        cfg.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"
        cfg.MODEL.RPN_ONLY = True
        cfg.MODEL.FCOS_ON = True
        cfg.MODEL.RETINANET_ON = False
        cfg.MODEL.BACKBONE.CONV_BODY = "R-50-FPN-RETINANET"
        cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS = 256
        cfg.MODEL.RETINANET.USE_C5 = False
        cfg.MODEL.FCOS.NUM_CLASSES = int(self.num_classes_no_bg) + 1
        cfg.MODEL.FCOS.INFERENCE_TH = float(self.confidence)
        cfg.MODEL.FCOS.NMS_TH = float(self.iou_thresh)
        cfg.MODEL.FCOS.USE_DCN_IN_TOWER = False
        cfg.MODEL.DEVICE = str(self.device)
        cfg.freeze()
        return cfg

    def _build_model(self):
        from fcos_core.modeling.detector import build_detection_model

        model = build_detection_model(self.cfg)
        return model.to(self.device)

    def _build_preprocess_transforms(self):
        from fcos_core.data.transforms.transforms import Normalize, Resize

        resize = Resize(self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MAX_SIZE_TEST)
        normalize = Normalize(
            mean=self.cfg.INPUT.PIXEL_MEAN,
            std=self.cfg.INPUT.PIXEL_STD,
            to_bgr255=self.cfg.INPUT.TO_BGR255,
        )
        return resize, normalize

    def load_weights(self, model_weight):
        path = _ensure_fcos_weight(model_weight)
        state_dict = _strip_prefixes(_checkpoint_state_dict(_torch_load(path, self.device)))
        current = self.detector_model.state_dict()
        filtered = {
            key: value
            for key, value in state_dict.items()
            if key in current and tuple(current[key].shape) == tuple(value.shape)
        }
        skipped = len(state_dict) - len(filtered)
        if skipped:
            print(f"[WARN] FCOS checkpoint partial load: loaded={len(filtered)}, skipped={skipped}")
        self.detector_model.load_state_dict(filtered, strict=False)

    def get_resize_size(self, image_or_size):
        if isinstance(image_or_size, torch.Tensor):
            orig_h, orig_w = int(image_or_size.shape[-2]), int(image_or_size.shape[-1])
        else:
            orig_h, orig_w = int(image_or_size[0]), int(image_or_size[1])
        min_sizes = self._resize_transform.min_size
        size = int(min_sizes[0] if isinstance(min_sizes, (tuple, list)) else min_sizes)
        max_size = self._resize_transform.max_size
        min_original_size = float(min(orig_w, orig_h))
        max_original_size = float(max(orig_w, orig_h))
        if max_size is not None and max_original_size / min_original_size * size > max_size:
            size = int(round(max_size * min_original_size / max_original_size))
        if (orig_w <= orig_h and orig_w == size) or (orig_h <= orig_w and orig_h == size):
            return (orig_h, orig_w)
        if orig_w < orig_h:
            return (int(size * orig_h / orig_w), size)
        return (size, int(size * orig_w / orig_h))

    def get_resize_ratio(self, image_or_size):
        if isinstance(image_or_size, torch.Tensor):
            orig_h, orig_w = int(image_or_size.shape[-2]), int(image_or_size.shape[-1])
        else:
            orig_h, orig_w = int(image_or_size[0]), int(image_or_size[1])
        resized_h, resized_w = self.get_resize_size((orig_h, orig_w))
        return (float(resized_w) / float(max(orig_w, 1)), float(resized_h) / float(max(orig_h, 1)))

    def resize_image_for_display(self, image):
        resize_size = self.get_resize_size(image)
        resized = TF.resize(image.detach().cpu(), resize_size)
        return np.ascontiguousarray(np.clip(resized.numpy() * 255.0, 0, 255).astype(np.uint8))

    def preprocess_images(self, images):
        if isinstance(images, torch.Tensor):
            if images.dim() == 4:
                image_list = [img.to(self.device, dtype=torch.float32) for img in images]
            else:
                image_list = [images.to(self.device, dtype=torch.float32)]
        else:
            image_list = [img.to(self.device, dtype=torch.float32) for img in images]

        processed = []
        for img in image_list:
            orig_h, orig_w = int(img.shape[-2]), int(img.shape[-1])
            resize_size = self.get_resize_size((orig_h, orig_w))
            if (orig_h, orig_w) != tuple(resize_size):
                img = TF.resize(img, resize_size)
            img = self._normalize_transform(img)
            processed.append(img)
        return processed

    def forward(self, images):
        was_training = self.detector_model.training
        if self.mode != "train":
            self.detector_model.eval()
        with torch.no_grad():
            detections = self.detector_model(self.preprocess_images(images))
        if was_training and self.mode == "train":
            self.detector_model.train()
        return self._detections_to_contract(detections)

    def forward_preprocessed(self, processed_images):
        was_training = self.detector_model.training
        if self.mode != "train":
            self.detector_model.eval()
        with torch.no_grad():
            detections = self.detector_model(processed_images)
        if was_training and self.mode == "train":
            self.detector_model.train()
        return self._detections_to_contract(detections)

    def prepare_feature_cache(self, processed_images):
        from fcos_core.structures.image_list import to_image_list

        was_training = self.detector_model.training
        if self.mode != "train":
            self.detector_model.eval()
        with torch.no_grad():
            image_list = to_image_list(processed_images)
            features = self.detector_model.backbone(image_list.tensors)
        if was_training and self.mode == "train":
            self.detector_model.train()
        return {"images": image_list, "features": features}

    def forward_from_feature_cache(self, cache):
        was_training = self.detector_model.training
        if self.mode != "train":
            self.detector_model.eval()
        with torch.no_grad():
            detections, _losses = self.detector_model.rpn(
                cache["images"],
                cache["features"],
                None,
            )
        if was_training and self.mode == "train":
            self.detector_model.train()
        return self._detections_to_contract(detections)

    def _boxlists_to_contract(self, detections):
        rows_by_image = []
        logits_by_image = []
        has_logits = False
        num_classes = int(self.num_classes_no_bg)
        for det in detections:
            boxes = det.convert("xyxy").bbox.to(self.device)
            scores = det.get_field("scores").to(self.device) if det.has_field("scores") else boxes.new_zeros((boxes.shape[0],))
            labels_raw = det.get_field("labels").to(self.device).long() if det.has_field("labels") else boxes.new_zeros((boxes.shape[0],), dtype=torch.long)
            labels = (labels_raw - 1).clamp(min=0, max=max(num_classes - 1, 0))

            xywh = boxes.clone()
            if xywh.numel():
                xywh[:, 0] = (boxes[:, 0] + boxes[:, 2]) * 0.5
                xywh[:, 1] = (boxes[:, 1] + boxes[:, 3]) * 0.5
                xywh[:, 2] = (boxes[:, 2] - boxes[:, 0]).clamp(min=0.0)
                xywh[:, 3] = (boxes[:, 3] - boxes[:, 1]).clamp(min=0.0)

            if det.has_field("class_probs"):
                probs = det.get_field("class_probs").to(self.device, dtype=scores.dtype)
                if probs.shape[-1] != num_classes:
                    probs = probs[:, :num_classes] if probs.shape[-1] > num_classes else F.pad(probs, (0, num_classes - probs.shape[-1]))
            else:
                probs = torch.zeros((boxes.shape[0], num_classes), dtype=scores.dtype, device=self.device)
            if boxes.shape[0] > 0 and num_classes > 0 and not det.has_field("class_probs"):
                row_idx = torch.arange(boxes.shape[0], device=self.device)
                probs[row_idx, labels] = scores.clamp(min=0.0, max=1.0)
            if det.has_field("class_logits"):
                logits = det.get_field("class_logits").to(self.device, dtype=scores.dtype)
                if logits.shape[-1] != num_classes:
                    logits = logits[:, :num_classes] if logits.shape[-1] > num_classes else F.pad(logits, (0, num_classes - logits.shape[-1]))
                has_logits = True
            else:
                logits = torch.empty((boxes.shape[0], 0), dtype=scores.dtype, device=self.device)
            rows_by_image.append(torch.cat([xywh, scores[:, None], labels.to(scores.dtype)[:, None], probs], dim=1))
            logits_by_image.append(logits)
        return rows_by_image, logits_by_image if has_logits else None

    def _detections_to_contract(self, detections):
        return self._boxlists_to_contract(detections)

    def get_last_pre_nms_predictions(self):
        box_selector = getattr(getattr(self.detector_model, "rpn", None), "box_selector_test", None)
        pre_nms_boxlists = getattr(box_selector, "last_pre_nms_boxlists", None)
        if pre_nms_boxlists is None:
            return None, None
        return self._boxlists_to_contract(pre_nms_boxlists)

    def non_max_suppression(
        self,
        prediction,
        logits=None,
        conf_thres=0.05,
        iou_thres=0.6,
        classes=None,
        agnostic=False,
        max_det=None,
        return_indices=False,
        **_kwargs,
    ):
        outputs = []
        logits_outputs = []
        objectness_outputs = []
        index_outputs = []
        for image_idx, pred in enumerate(prediction):
            if pred.numel() == 0:
                device = pred.device
                logit_dim = 0
                if isinstance(logits, list) and image_idx < len(logits):
                    logit_dim = int(logits[image_idx].shape[-1])
                outputs.append(torch.zeros((0, 6), dtype=pred.dtype, device=device))
                logits_outputs.append(torch.zeros((0, logit_dim), dtype=pred.dtype, device=device))
                objectness_outputs.append(torch.zeros((0, 1), dtype=pred.dtype, device=device))
                index_outputs.append(torch.zeros((0,), dtype=torch.long, device=device))
                continue

            scores = pred[:, 4]
            labels = pred[:, 5].long()
            # FCOSPostProcessor already applies score filtering, top-k, and class-wise NMS.
            # Keep this wrapper as a lightweight compatibility adapter: optional class
            # filtering, max_det truncation, xywh->xyxy conversion, and index reporting.
            keep = scores > float(conf_thres)
            if classes is not None:
                class_tensor = torch.as_tensor(classes, device=pred.device, dtype=torch.long)
                keep &= (labels[:, None] == class_tensor[None]).any(dim=1)
            keep_idx = torch.nonzero(keep, as_tuple=False).flatten()
            if max_det is not None and keep_idx.numel() > int(max_det):
                keep_idx = keep_idx[: int(max_det)]

            xywh = pred[keep_idx, :4]
            xyxy = xywh.clone()
            if xyxy.numel():
                xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] * 0.5
                xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] * 0.5
                xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] * 0.5
                xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] * 0.5
            det = torch.cat([xyxy, scores[keep_idx, None], labels[keep_idx, None].to(pred.dtype)], dim=1)
            outputs.append(det)
            if isinstance(logits, list) and image_idx < len(logits):
                logits_outputs.append(logits[image_idx][keep_idx])
            else:
                logits_outputs.append(pred[keep_idx, 6:])
            objectness_outputs.append(torch.ones((keep_idx.shape[0], 1), dtype=pred.dtype, device=pred.device))
            index_outputs.append(keep_idx)

        if return_indices:
            return outputs, logits_outputs, objectness_outputs, index_outputs
        return outputs, logits_outputs, objectness_outputs

    def set_dropout_rate(self, dropout_rate):
        for handle in self._dropout_handles:
            handle.remove()
        self._dropout_handles = []
        self._dropout_rate = float(dropout_rate)
        if self._dropout_rate <= 0.0:
            return

        def _dropout_hook(_module, _inputs, output):
            return F.dropout2d(output, p=self._dropout_rate, training=True)

        head = getattr(getattr(self.detector_model, "rpn", None), "head", None)
        for tower_name in ("cls_tower", "bbox_tower"):
            tower = getattr(head, tower_name, None)
            if tower is None:
                continue
            for module in tower.modules():
                if isinstance(module, nn.ReLU):
                    self._dropout_handles.append(module.register_forward_hook(_dropout_hook))
