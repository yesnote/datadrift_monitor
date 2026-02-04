from __future__ import annotations

import torch
from typing import List, Optional


class YOLO26LP:
    """
    YOLO26-LP wrapper for DiL (NO NMS).

    Provides:
      1) Detect head raw output (Mo)          -> for Grad-CAM (Eq.1)
      2) Confidence-thresholded boxes only    -> for BL / CL computation
    """

    def __init__(
        self,
        model: torch.nn.Module,
        conf_thresh: float = 0.5,
        device: Optional[str] = None,
        img_size: int = 640,
    ):
        self.model = model
        self.conf_thresh = conf_thresh
        self.img_size = img_size
        self.device = device or next(model.parameters()).device

        self.model.eval().to(self.device)

        # Detect head raw output (Mo)
        self._det_raw = None
        self._register_detect_hook()

    # --------------------------------------------------
    # Detect head hook (NMS 이전)
    # --------------------------------------------------
    def _register_detect_hook(self):
        """
        YOLO26: Detect head = last module
        """
        detect_layer = self.model.model[-1]

        def hook_fn(module, inputs, outputs):
            # outputs: raw detect tensor(s), NMS 이전
            self._det_raw = outputs

        detect_layer.register_forward_hook(hook_fn)

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------
    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (1, 3, H, W), float in [0,1]

        Returns:
            det_raw   : Detect head raw output (Mo)
            pred_boxes: List[[x1,y1,x2,y2]]  (conf threshold only)
        """
        x = x.to(self.device)

        # 1. forward (hook에서 det_raw 저장됨)
        outputs = self.model(x)

        if self._det_raw is None:
            raise RuntimeError("Detect head raw output was not captured.")

        # 2. decode boxes from detect output (NO NMS)
        pred_boxes = self._decode_boxes(self._det_raw, x.shape[-2:])

        return self._det_raw, pred_boxes

    # --------------------------------------------------
    # Decode YOLO26 detect output (NO NMS)
    # --------------------------------------------------
    def _decode_boxes(self, det_raw, img_hw):
        """
        det_raw: list of feature maps from detect head
        img_hw : (H, W)

        YOLO-style output per scale:
          (B, A, H, W, 5 + C)
          -> [cx, cy, w, h, obj, cls...]
        """
        H_img, W_img = img_hw
        boxes = []

        if not isinstance(det_raw, (list, tuple)):
            det_raw = [det_raw]

        for out in det_raw:
            # expected shape: (B, A, H, W, 5 + C)
            if out.dim() != 5:
                continue

            out = out[0]  # batch=1
            obj = out[..., 4]

            mask = obj >= self.conf_thresh
            if mask.sum() == 0:
                continue

            cx = out[..., 0][mask]
            cy = out[..., 1][mask]
            w  = out[..., 2][mask]
            h  = out[..., 3][mask]

            # assume cx,cy,w,h are normalized (YOLO-style)
            x1 = (cx - w / 2) * W_img
            y1 = (cy - h / 2) * H_img
            x2 = (cx + w / 2) * W_img
            y2 = (cy + h / 2) * H_img

            for b in zip(x1, y1, x2, y2):
                boxes.append([v.item() for v in b])

        return boxes
