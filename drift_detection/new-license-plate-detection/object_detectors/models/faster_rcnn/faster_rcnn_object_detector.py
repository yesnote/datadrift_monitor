from pathlib import Path
from urllib.parse import urlparse
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import batched_nms, boxes as box_ops


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FASTER_RCNN_COCO_WEIGHT = (
    PROJECT_ROOT / "models" / "faster_rcnn" / "weights" / "coco" / "fasterrcnn_resnet50_fpn_coco.pth"
)


def _torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _is_default_coco_weight(path):
    if path is None:
        return True
    return Path(path).name == DEFAULT_FASTER_RCNN_COCO_WEIGHT.name


def _seed_torchvision_cache_from_local(local_path):
    local_path = Path(local_path)
    if not local_path.is_file():
        return
    filename = Path(urlparse(FasterRCNN_ResNet50_FPN_Weights.DEFAULT.url).path).name
    cache_path = Path(torch.hub.get_dir()) / "checkpoints" / filename
    if cache_path.is_file():
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local_path, cache_path)


def _real_coco_categories():
    categories = list(FasterRCNN_ResNet50_FPN_Weights.DEFAULT.meta.get("categories", []))
    return [name for name in categories if name not in {"__background__", "N/A"}]


def _load_checkpoint_state(model, checkpoint_path, device):
    payload = _torch_load(checkpoint_path, map_location=device)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
    elif isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], nn.Module):
        state_dict = payload["model"].state_dict()
    elif isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    elif isinstance(payload, dict):
        state_dict = payload
    else:
        raise ValueError(f"Unsupported Faster R-CNN checkpoint payload: {checkpoint_path}")

    current = model.state_dict()
    filtered = {
        key: value
        for key, value in state_dict.items()
        if key in current and tuple(current[key].shape) == tuple(value.shape)
    }
    missing = len(current) - len(filtered)
    if missing:
        print(f"[WARN] Faster R-CNN checkpoint partial load: loaded={len(filtered)}, skipped={missing}")
    model.load_state_dict(filtered, strict=False)


class _DropoutFastRCNNPredictor(nn.Module):
    def __init__(self, base_predictor, dropout_rate=0.0):
        super().__init__()
        in_features = base_predictor.cls_score.in_features
        num_classes = base_predictor.cls_score.out_features
        self.dropout = float(dropout_rate)
        self.cls_score = nn.Linear(in_features, num_classes)
        self.bbox_pred = nn.Linear(in_features, num_classes * 4)
        self.cls_score.load_state_dict(base_predictor.cls_score.state_dict())
        self.bbox_pred.load_state_dict(base_predictor.bbox_pred.state_dict())

    def forward(self, x):
        if x.dim() == 4:
            x = x.flatten(start_dim=1)
        if self.dropout > 0.0:
            x = F.dropout(x, p=self.dropout, training=True)
        return self.cls_score(x), self.bbox_pred(x)


class _ForwardProxy:
    def __init__(self, owner):
        self._owner = owner

    def __call__(self, images, augment=False, need_logits=True, **_kwargs):
        return self._owner._forward_impl(images, need_logits=need_logits)

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
        transform_min_size=None,
        transform_max_size=None,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.img_size = img_size if isinstance(img_size, tuple) else (int(img_size), int(img_size))
        self.transform_min_size = int(transform_min_size) if transform_min_size is not None else int(min(self.img_size))
        self.transform_max_size = int(transform_max_size) if transform_max_size is not None else int(max(self.img_size))
        self.mode = str(mode)
        self.confidence = float(confidence)
        self.conf_thresh = float(confidence)
        self.iou_thresh = float(iou_thresh)
        self.max_det = int(max_det)
        self.agnostic = False
        self.agnostic_nms = False
        self.is_faster_rcnn = True
        self.has_faster_rcnn_label_column = True
        self.uses_custom_faster_rcnn_forward = True

        self.names = list(names or _real_coco_categories())
        self.num_classes_no_bg = len(self.names)
        self._torchvision_categories = list(FasterRCNN_ResNet50_FPN_Weights.DEFAULT.meta.get("categories", []))
        self._uses_torchvision_coco_space = self.names == _real_coco_categories()
        self._output_class_ids = self._build_output_class_ids()
        self.num_classes_with_bg = (
            len(self._torchvision_categories)
            if self._uses_torchvision_coco_space and bool(pretrained) and _is_default_coco_weight(model_weight)
            else self.num_classes_no_bg + 1
        )

        self.detector_model = self._build_model(model_weight=model_weight, pretrained=bool(pretrained))
        self.model = _ForwardProxy(self)
        self.detector_model.to(self.device)
        self.detector_model.train() if self.mode == "train" else self.detector_model.eval()
        print("[INFO] Faster R-CNN model is loaded")

    def _build_output_class_ids(self):
        if not self._uses_torchvision_coco_space:
            return list(range(1, self.num_classes_no_bg + 1))
        name_to_idx = {
            str(name): idx
            for idx, name in enumerate(self._torchvision_categories)
            if name not in {"__background__", "N/A"}
        }
        return [int(name_to_idx[name]) for name in self.names]

    def _build_model(self, model_weight, pretrained):
        use_official_coco = pretrained and self._uses_torchvision_coco_space and _is_default_coco_weight(model_weight)
        if use_official_coco and model_weight is not None:
            _seed_torchvision_cache_from_local(model_weight)
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if use_official_coco else None
        model = fasterrcnn_resnet50_fpn(
            weights=weights,
            weights_backbone=None,
            num_classes=None if weights is not None else self.num_classes_with_bg,
            min_size=self.transform_min_size,
            max_size=self.transform_max_size,
            box_score_thresh=0.0,
            box_nms_thresh=self.iou_thresh,
            box_detections_per_img=self.max_det,
        )

        if not use_official_coco:
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes_with_bg)

        # Keep the downloaded local COCO file as a cache artifact, but do not load it manually
        # for the default COCO case. The official weights path builds the matching FrozenBN model.
        if model_weight and Path(model_weight).is_file() and not use_official_coco:
            _load_checkpoint_state(model, Path(model_weight), self.device)
        elif model_weight and not Path(model_weight).is_file() and not use_official_coco:
            raise FileNotFoundError(f"Faster R-CNN weights not found: {model_weight}")

        return model

    def set_dropout_rate(self, dropout_rate):
        predictor = self.detector_model.roi_heads.box_predictor
        if not isinstance(predictor, _DropoutFastRCNNPredictor):
            self.detector_model.roi_heads.box_predictor = _DropoutFastRCNNPredictor(predictor, dropout_rate=0.0)
            predictor = self.detector_model.roi_heads.box_predictor
        predictor.dropout = float(dropout_rate)

    def _map_labels_to_output(self, labels_internal):
        labels_internal = labels_internal.to(torch.long)
        if self._uses_torchvision_coco_space and self.num_classes_with_bg == len(self._torchvision_categories):
            internal_to_output = torch.full(
                (len(self._torchvision_categories),),
                -1,
                dtype=torch.long,
                device=labels_internal.device,
            )
            for out_idx, internal_idx in enumerate(self._output_class_ids):
                internal_to_output[int(internal_idx)] = int(out_idx)
            valid = (labels_internal >= 0) & (labels_internal < internal_to_output.numel())
            labels = internal_to_output[labels_internal.clamp(min=0, max=internal_to_output.numel() - 1)]
            valid &= labels >= 0
            return labels, valid

        labels = labels_internal - 1
        valid = (labels >= 0) & (labels < self.num_classes_no_bg)
        return labels, valid

    def _forward_impl(self, images, need_logits=True):
        if isinstance(images, torch.Tensor):
            image_list = [img for img in images]
        else:
            image_list = list(images)

        was_training = self.detector_model.training
        self.detector_model.eval()
        with torch.inference_mode():
            image_list = [
                img.to(self.device, non_blocking=True) if img.device != self.device else img
                for img in image_list
            ]
            if need_logits:
                detections = self._custom_inference(image_list)
            else:
                detections = self.detector_model(image_list)
        if was_training:
            self.detector_model.train()
        return self._detections_to_contract(detections, self.device)

    def _custom_inference(self, image_list):
        """Run Faster R-CNN explicitly so uncertainty code can use ROI logits.

        torchvision's public eval forward only returns postprocessed boxes,
        labels, and scores.  For UQ we also need the per-detection class
        logits/probability vector, so we call the same internal stages and keep
        the selected ROI logits after class-wise NMS.
        """
        original_image_sizes = [(int(img.shape[-2]), int(img.shape[-1])) for img in image_list]
        images, _targets = self.detector_model.transform(image_list, None)

        features = self.detector_model.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = {"0": features}

        proposals, _proposal_losses = self.detector_model.rpn(images, features, None)
        roi_heads = self.detector_model.roi_heads
        box_features = roi_heads.box_roi_pool(features, proposals, images.image_sizes)
        box_features = roi_heads.box_head(box_features)
        class_logits, box_regression = roi_heads.box_predictor(box_features)

        detections = self._postprocess_detections_with_logits(
            class_logits=class_logits,
            box_regression=box_regression,
            proposals=proposals,
            image_shapes=images.image_sizes,
        )
        detections = self.detector_model.transform.postprocess(detections, images.image_sizes, original_image_sizes)
        return detections

    def _postprocess_detections_with_logits(self, class_logits, box_regression, proposals, image_shapes):
        device = class_logits.device
        num_classes = int(class_logits.shape[-1])
        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.detector_model.roi_heads.box_coder.decode(box_regression, proposals)
        pred_scores = F.softmax(class_logits, -1)
        pred_logits = class_logits.split(boxes_per_image, 0)
        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_detections = []
        for boxes, scores, logits, image_shape in zip(pred_boxes_list, pred_scores_list, pred_logits, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            num_proposals = int(scores.shape[0])
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)
            proposal_indices = torch.arange(num_proposals, device=device).view(-1, 1).expand_as(scores)

            # Remove background class, then flatten proposal x class candidates.
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]
            proposal_indices = proposal_indices[:, 1:]

            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)
            proposal_indices = proposal_indices.reshape(-1)

            inds = torch.where(scores > float(self.detector_model.roi_heads.score_thresh))[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]
            proposal_indices = proposal_indices[inds]

            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            proposal_indices = proposal_indices[keep]

            keep = batched_nms(boxes, scores, labels, float(self.detector_model.roi_heads.nms_thresh))
            keep = keep[: int(self.detector_model.roi_heads.detections_per_img)]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            logits = logits[proposal_indices[keep]]

            all_detections.append({
                "boxes": boxes,
                "labels": labels,
                "scores": scores,
                "class_logits": logits,
            })
        return all_detections

    def _select_output_class_columns(self, values):
        if self._uses_torchvision_coco_space and values.shape[-1] == len(self._torchvision_categories):
            return values[..., self._output_class_ids]
        if values.shape[-1] == self.num_classes_no_bg + 1:
            return values[..., 1:]
        return values[..., : self.num_classes_no_bg]

    def _detections_to_contract(self, detections, device):
        rows_by_image = []
        logits_by_image = []
        c = self.num_classes_no_bg

        for det in detections:
            boxes = det["boxes"].to(device)
            scores = det["scores"].to(device)
            labels_internal = det["labels"].to(device)
            labels, valid = self._map_labels_to_output(labels_internal)

            valid &= scores > 0.0
            boxes = boxes[valid][: self.max_det]
            scores = scores[valid][: self.max_det]
            labels = labels[valid][: self.max_det]

            raw_logits = det.get("class_logits")
            if raw_logits is not None:
                raw_logits = raw_logits.to(device)
                logits_full = self._select_output_class_columns(raw_logits)
                probs_full = F.softmax(logits_full, dim=-1)
                logits = torch.zeros((boxes.shape[0], c), dtype=scores.dtype, device=device)
                probs = torch.zeros((boxes.shape[0], c), dtype=scores.dtype, device=device)
                if int(valid.sum().item()) > 0:
                    logits = logits_full[valid][: self.max_det]
                    probs = probs_full[valid][: self.max_det]
            else:
                probs = torch.zeros((boxes.shape[0], c), dtype=scores.dtype, device=device)
                logits = torch.zeros((boxes.shape[0], c), dtype=scores.dtype, device=device)
            if boxes.shape[0] > 0:
                if raw_logits is None:
                    idx = torch.arange(boxes.shape[0], device=device)
                    probs[idx, labels] = scores
                    logits[idx, labels] = torch.logit(scores.clamp(min=1e-6, max=1.0 - 1e-6))

            xywh = boxes.clone()
            if xywh.numel():
                xywh[:, 0] = (boxes[:, 0] + boxes[:, 2]) * 0.5
                xywh[:, 1] = (boxes[:, 1] + boxes[:, 3]) * 0.5
                xywh[:, 2] = (boxes[:, 2] - boxes[:, 0]).clamp(min=0.0)
                xywh[:, 3] = (boxes[:, 3] - boxes[:, 1]).clamp(min=0.0)

            rows = torch.cat([xywh, scores[:, None], labels.to(scores.dtype)[:, None], probs], dim=1)
            if rows.shape[0] < self.max_det:
                pad = self.max_det - rows.shape[0]
                rows = torch.cat([rows, torch.zeros((pad, 6 + c), dtype=rows.dtype, device=device)], dim=0)
                logits = torch.cat([logits, torch.zeros((pad, c), dtype=logits.dtype, device=device)], dim=0)
            rows_by_image.append(rows)
            logits_by_image.append(logits)

        if not rows_by_image:
            return (
                torch.zeros((0, self.max_det, 6 + c), dtype=torch.float32, device=device),
                torch.zeros((0, self.max_det, c), dtype=torch.float32, device=device),
            )
        return torch.stack(rows_by_image, dim=0), torch.stack(logits_by_image, dim=0)

    def forward(self, images, augment=False):
        return self._forward_impl(images)

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
        for image_idx, pred in enumerate(prediction):
            if pred.numel() == 0:
                device = prediction.device
                outputs.append(torch.zeros((0, 6), dtype=prediction.dtype, device=device))
                logits_outputs.append(torch.zeros((0, logits.shape[-1]), dtype=prediction.dtype, device=device))
                objectness_outputs.append(torch.zeros((0, 1), dtype=prediction.dtype, device=device))
                index_outputs.append(torch.zeros((0,), dtype=torch.long, device=device))
                continue

            scores = pred[:, 4]
            labels = pred[:, 5].to(torch.long)
            keep = scores > float(conf_thres)
            if classes is not None:
                class_tensor = torch.as_tensor(classes, device=pred.device, dtype=torch.long)
                keep &= (labels[:, None] == class_tensor[None]).any(dim=1)
            keep_idx = torch.nonzero(keep, as_tuple=False).flatten()[: int(max_det)]

            xywh = pred[keep_idx, :4]
            xyxy = xywh.clone()
            if xyxy.numel():
                xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] * 0.5
                xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] * 0.5
                xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] * 0.5
                xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] * 0.5
            det = torch.cat([xyxy, scores[keep_idx, None], labels[keep_idx, None].to(pred.dtype)], dim=1)
            outputs.append(det)
            logits_outputs.append(logits[image_idx][keep_idx] if logits is not None else pred[keep_idx, 6:])
            objectness_outputs.append(torch.ones((keep_idx.shape[0], 1), dtype=pred.dtype, device=pred.device))
            index_outputs.append(keep_idx)

        if return_indices:
            return outputs, logits_outputs, objectness_outputs, index_outputs
        return outputs, logits_outputs, objectness_outputs
