from pathlib import Path
import sys

import torch
import torch.nn as nn


FCOS_PACKAGE_ROOT = Path(__file__).resolve().parent
if str(FCOS_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(FCOS_PACKAGE_ROOT))


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
        config_file=None,
        max_detections=100,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.mode = str(mode)
        self.confidence = float(confidence)
        self.conf_thresh = float(confidence)
        self.iou_thresh = float(iou_thresh)
        self.max_det = int(max_detections)
        self.agnostic = False
        self.agnostic_nms = False
        self.is_fcos = True

        self.names = list(names or [])
        self.num_classes_no_bg = len(self.names) if self.names else 80

        self.cfg = self._build_cfg(config_file)
        self.detector_model = self._build_model()
        self.model = _ForwardProxy(self)

        if model_weight:
            self.load_weights(model_weight)
        self.to(self.device)
        if self.mode == "train":
            self.train()
        else:
            self.eval()

    def _build_cfg(self, config_file=None):
        try:
            from fcos_core.config import cfg as base_cfg
        except ImportError as exc:
            raise ImportError(
                "FCOS config requires yacs and the copied fcos_core package. "
                "Install the FCOS reference dependencies if this import fails."
            ) from exc

        cfg = base_cfg.clone()
        if config_file:
            config_path = Path(config_file)
            if config_path.is_file():
                cfg.merge_from_file(str(config_path))

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
        cfg.TEST.DETECTIONS_PER_IMG = int(self.max_det)
        cfg.MODEL.DEVICE = str(self.device)
        cfg.freeze()
        return cfg

    def _build_model(self):
        from fcos_core.modeling.detector import build_detection_model

        model = build_detection_model(self.cfg)
        return model.to(self.device)

    def load_weights(self, model_weight):
        path = Path(model_weight)
        if not path.is_file():
            raise FileNotFoundError(f"FCOS weight file not found: {path}")
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

    def forward(self, images):
        if isinstance(images, torch.Tensor):
            if images.dim() == 4:
                image_list = [img.to(self.device) for img in images]
            else:
                image_list = [images.to(self.device)]
        else:
            image_list = [img.to(self.device) for img in images]
        return self.detector_model(image_list)
