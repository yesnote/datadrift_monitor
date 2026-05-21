from pathlib import Path
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import batched_nms, clip_boxes_to_image, remove_small_boxes

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FASTER_RCNN_COCO_WEIGHT = (
    PROJECT_ROOT / "models" / "faster_rcnn" / "weights" / "coco" / "fasterrcnn_resnet50_fpn_coco.pth"
)


def _letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = (r, r)
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    if auto:
        dw = np.mod(dw, stride)
        dh = np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = (new_shape[1] / shape[1], new_shape[0] / shape[0])

    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)


def _torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _ensure_default_coco_weight(path: Path = DEFAULT_FASTER_RCNN_COCO_WEIGHT) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    torch.hub.download_url_to_file(weights.url, str(path), progress=True)
    return path


def _load_matching_state_dict(model: nn.Module, state_dict: dict) -> None:
    current = model.state_dict()
    filtered = {
        key: value
        for key, value in state_dict.items()
        if key in current and tuple(current[key].shape) == tuple(value.shape)
    }
    model.load_state_dict(filtered, strict=False)


def _real_coco_categories():
    categories = list(FasterRCNN_ResNet50_FPN_Weights.DEFAULT.meta.get("categories", []))
    return [name for name in categories if name not in {"__background__", "N/A"}]


class _DropoutFastRCNNPredictor(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, dropout_rate: float = 0.0):
        super().__init__()
        self.dropout = float(dropout_rate)
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            if list(x.shape[2:]) != [1, 1]:
                raise AssertionError("Fast R-CNN predictor expects pooled features with spatial size 1x1.")
            x = x.flatten(start_dim=1)
        if self.dropout > 0.0:
            x = F.dropout(x, p=float(self.dropout), training=True)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas


class _FasterRCNNForwardProxy:
    def __init__(self, owner):
        object.__setattr__(self, "_owner", owner)

    def __call__(self, img, augment=False, **_kwargs):
        return self._owner._forward_impl(img)

    def __getattr__(self, name):
        return getattr(self._owner.detector_model, name)


class FasterRCNNTorchObjectDetector(nn.Module):
    def __init__(
        self,
        model_weight=None,
        device="cuda",
        img_size=(640, 640),
        names=None,
        mode="eval",
        confidence=0.25,
        iou_thresh=0.45,
        max_det=300,
        pretrained=True,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.img_size = img_size if isinstance(img_size, tuple) else (int(img_size), int(img_size))
        self.mode = str(mode)
        self.confidence = float(confidence)
        self.conf_thresh = float(confidence)
        self.iou_thresh = float(iou_thresh)
        self.agnostic = False
        self.agnostic_nms = False
        self.max_det = int(max_det)
        self.is_faster_rcnn = True
        self.names = list(names or self._default_coco_names())
        self.num_classes_no_bg = int(len(self.names))
        self._torchvision_categories = list(FasterRCNN_ResNet50_FPN_Weights.DEFAULT.meta.get("categories", []))
        self._uses_torchvision_coco_space = self.names == _real_coco_categories()
        self._output_class_ids = self._build_output_class_ids()
        self.num_classes_with_bg = (
            len(self._torchvision_categories)
            if self._uses_torchvision_coco_space
            else self.num_classes_no_bg + 1
        )
        self.detector_model = self._build_model(model_weight, bool(pretrained))
        self.model = _FasterRCNNForwardProxy(self)
        self.detector_model.to(self.device)
        self.detector_model.train() if self.mode == "train" else self.detector_model.eval()
        print("[INFO] Faster R-CNN model is loaded")

    @staticmethod
    def _default_coco_names():
        return _real_coco_categories()

    def _build_output_class_ids(self):
        if not self._uses_torchvision_coco_space:
            return list(range(1, self.num_classes_no_bg + 1))
        name_to_idx = {
            str(name): idx
            for idx, name in enumerate(self._torchvision_categories)
            if name not in {"__background__", "N/A"}
        }
        missing = [name for name in self.names if name not in name_to_idx]
        if missing:
            raise ValueError(f"COCO class names missing from torchvision categories: {missing}")
        return [int(name_to_idx[name]) for name in self.names]

    @staticmethod
    def yolo_resize(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
        return _letterbox(img, new_shape=new_shape, color=color, auto=auto, scaleFill=scaleFill, scaleup=scaleup)

    def _replace_predictor(self, model, dropout_rate=0.0):
        old_predictor = model.roi_heads.box_predictor
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        new_predictor = _DropoutFastRCNNPredictor(
            in_channels=in_features,
            num_classes=self.num_classes_with_bg,
            dropout_rate=dropout_rate,
        )
        if int(old_predictor.cls_score.out_features) == self.num_classes_with_bg:
            new_predictor.cls_score.load_state_dict(old_predictor.cls_score.state_dict())
            new_predictor.bbox_pred.load_state_dict(old_predictor.bbox_pred.state_dict())
        model.roi_heads.box_predictor = new_predictor
        return model

    def _build_model(self, model_weight, pretrained):
        weights = None
        model = fasterrcnn_resnet50_fpn(
            weights=weights,
            weights_backbone=None if weights is None else None,
            min_size=int(self.img_size[0]),
            max_size=int(max(self.img_size)),
            box_score_thresh=0.0,
            box_nms_thresh=self.iou_thresh,
            box_detections_per_img=self.max_det,
        )

        current_classes = int(model.roi_heads.box_predictor.cls_score.out_features)
        if current_classes != self.num_classes_with_bg or not isinstance(model.roi_heads.box_predictor, _DropoutFastRCNNPredictor):
            self._replace_predictor(model, dropout_rate=0.0)

        if model_weight:
            path = Path(model_weight)
            if path.is_file():
                payload = _torch_load(path, map_location=self.device)
                if isinstance(payload, dict) and "model_state_dict" in payload:
                    state_dict = payload["model_state_dict"]
                elif isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], nn.Module):
                    state_dict = payload["model"].state_dict()
                elif isinstance(payload, dict) and "state_dict" in payload:
                    state_dict = payload["state_dict"]
                elif isinstance(payload, dict):
                    state_dict = payload
                else:
                    raise ValueError(f"Unsupported Faster R-CNN checkpoint payload: {path}")
                _load_matching_state_dict(model, state_dict)
            elif pretrained and path.name == DEFAULT_FASTER_RCNN_COCO_WEIGHT.name:
                state_dict = _torch_load(_ensure_default_coco_weight(path), map_location=self.device)
                _load_matching_state_dict(model, state_dict)
            else:
                raise FileNotFoundError(f"Faster R-CNN weights not found: {path}")
        elif pretrained:
            path = _ensure_default_coco_weight()
            state_dict = _torch_load(path, map_location=self.device)
            _load_matching_state_dict(model, state_dict)
        return model

    def set_dropout_rate(self, dropout_rate: float):
        predictor = self.detector_model.roi_heads.box_predictor
        if hasattr(predictor, "dropout"):
            predictor.dropout = float(dropout_rate)

    def _forward_impl(self, images_tensor: torch.Tensor):
        images_list = [img.to(self.device) for img in images_tensor]
        original_image_sizes = [tuple(img.shape[-2:]) for img in images_list]
        images, _ = self.detector_model.transform(images_list, None)
        features = self.detector_model.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        proposals, _proposal_losses = self.detector_model.rpn(images, features, None)
        box_features = self.detector_model.roi_heads.box_roi_pool(features, proposals, images.image_sizes)
        box_features = self.detector_model.roi_heads.box_head(box_features)
        class_logits, box_regression = self.detector_model.roi_heads.box_predictor(box_features)
        raw_prediction, raw_logits = self._postprocess_to_yolo_contract(
            class_logits=class_logits,
            box_regression=box_regression,
            proposals=proposals,
            image_shapes=images.image_sizes,
            original_image_sizes=original_image_sizes,
            device=images_tensor.device,
        )
        return raw_prediction, raw_logits

    def _postprocess_to_yolo_contract(self, class_logits, box_regression, proposals, image_shapes, original_image_sizes, device):
        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.detector_model.roi_heads.box_coder.decode(box_regression, proposals)
        pred_scores = F.softmax(class_logits, dim=-1)
        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)
        pred_logits_list = class_logits.split(boxes_per_image, 0)

        batch_rows = []
        batch_logits = []
        c = self.num_classes_no_bg
        class_ids = torch.as_tensor(self._output_class_ids, dtype=torch.long, device=class_logits.device)
        for boxes, scores, logits, image_shape, original_size in zip(
            pred_boxes_list, pred_scores_list, pred_logits_list, image_shapes, original_image_sizes
        ):
            boxes = clip_boxes_to_image(boxes, image_shape)
            boxes = boxes[:, class_ids]
            scores_no_bg = scores[:, class_ids]
            logits_no_bg = logits[:, class_ids]
            labels = torch.arange(c, device=scores.device).view(1, -1).expand_as(scores_no_bg)

            boxes = boxes.reshape(-1, 4)
            scores_flat = scores_no_bg.reshape(-1)
            labels = labels.reshape(-1)
            proposal_idx = torch.arange(scores_no_bg.shape[0], device=scores.device).view(-1, 1).expand_as(scores_no_bg).reshape(-1)

            keep = scores_flat > 0.0
            boxes, scores_flat, labels, proposal_idx = boxes[keep], scores_flat[keep], labels[keep], proposal_idx[keep]
            keep = remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores_flat, labels, proposal_idx = boxes[keep], scores_flat[keep], labels[keep], proposal_idx[keep]
            keep = batched_nms(boxes, scores_flat, labels, self.iou_thresh)
            keep = keep[: self.max_det]
            boxes, scores_flat, labels, proposal_idx = boxes[keep], scores_flat[keep], labels[keep], proposal_idx[keep]

            result = [{"boxes": boxes, "scores": scores_flat, "labels": labels}]
            result = self.detector_model.transform.postprocess(result, [image_shape], [original_size])[0]
            boxes = result["boxes"]
            scores_flat = result["scores"]

            probs = torch.zeros((boxes.shape[0], c), dtype=scores_flat.dtype, device=scores_flat.device)
            logits_sel = torch.zeros((boxes.shape[0], c), dtype=logits.dtype, device=logits.device)
            if boxes.shape[0] > 0:
                probs_all = scores_no_bg[proposal_idx]
                logits_all = logits_no_bg[proposal_idx]
                probs = probs_all[: boxes.shape[0]]
                logits_sel = logits_all[: boxes.shape[0]]

            xywh = boxes.clone()
            xywh[:, 0] = (boxes[:, 0] + boxes[:, 2]) * 0.5
            xywh[:, 1] = (boxes[:, 1] + boxes[:, 3]) * 0.5
            xywh[:, 2] = (boxes[:, 2] - boxes[:, 0]).clamp(min=0.0)
            xywh[:, 3] = (boxes[:, 3] - boxes[:, 1]).clamp(min=0.0)
            rows = torch.cat([xywh, scores_flat.unsqueeze(1), probs], dim=1)
            if rows.shape[0] < self.max_det:
                pad_n = self.max_det - rows.shape[0]
                rows = torch.cat([rows, torch.zeros((pad_n, 5 + c), dtype=rows.dtype, device=rows.device)], dim=0)
                logits_sel = torch.cat([logits_sel, torch.zeros((pad_n, c), dtype=logits_sel.dtype, device=logits_sel.device)], dim=0)
            else:
                rows = rows[: self.max_det]
                logits_sel = logits_sel[: self.max_det]
            batch_rows.append(rows.to(device))
            batch_logits.append(logits_sel.to(device))

        if not batch_rows:
            return (
                torch.zeros((0, self.max_det, 5 + c), dtype=torch.float32, device=device),
                torch.zeros((0, self.max_det, c), dtype=torch.float32, device=device),
            )
        return torch.stack(batch_rows, dim=0), torch.stack(batch_logits, dim=0)

    def forward(self, img, augment=False):
        return self._forward_impl(img)

    @staticmethod
    def non_max_suppression(
        prediction,
        logits,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        max_det=300,
        return_indices=False,
        **_kwargs,
    ):
        outputs = []
        logits_outputs = []
        objectness_outputs = []
        index_outputs = []
        for xi, x in enumerate(prediction):
            if x.numel() == 0:
                device = prediction.device
                outputs.append(torch.zeros((0, 6), dtype=prediction.dtype, device=device))
                logits_outputs.append(torch.zeros((0, logits.shape[-1]), dtype=prediction.dtype, device=device))
                objectness_outputs.append(torch.zeros((0, 1), dtype=prediction.dtype, device=device))
                index_outputs.append(torch.zeros((0,), dtype=torch.long, device=device))
                continue
            score = x[:, 4]
            cls_prob = x[:, 5:]
            cls_score, cls_idx = cls_prob.max(dim=1) if cls_prob.numel() else (score, torch.zeros_like(score, dtype=torch.long))
            keep = score > float(conf_thres)
            if classes is not None:
                class_tensor = torch.tensor(classes, device=x.device, dtype=torch.long)
                keep &= (cls_idx[:, None] == class_tensor[None]).any(dim=1)
            keep_idx = torch.nonzero(keep, as_tuple=False).flatten()
            keep_idx = keep_idx[: int(max_det)]
            xywh = x[keep_idx, :4]
            xyxy = xywh.clone()
            xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] * 0.5
            xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] * 0.5
            xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] * 0.5
            xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] * 0.5
            det = torch.cat([xyxy, score[keep_idx, None], cls_idx[keep_idx, None].float()], dim=1)
            outputs.append(det)
            logits_outputs.append(logits[xi][keep_idx] if logits is not None else cls_prob[keep_idx])
            objectness_outputs.append(torch.ones((keep_idx.shape[0], 1), dtype=prediction.dtype, device=prediction.device))
            index_outputs.append(keep_idx)
        if return_indices:
            return outputs, logits_outputs, objectness_outputs, index_outputs
        return outputs, logits_outputs, objectness_outputs

    def preprocessing(self, img):
        if len(img.shape) != 4:
            img = np.expand_dims(img, axis=0)
        im0 = img.astype(np.uint8)
        img = np.array([self.yolo_resize(im, new_shape=self.img_size, auto=False)[0] for im in im0])
        img = img.transpose((0, 3, 1, 2))
        img = np.ascontiguousarray(img)
        return torch.from_numpy(img).to(self.device).float() / 255.0
