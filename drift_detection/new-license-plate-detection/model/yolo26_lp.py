from __future__ import annotations

import torch
from typing import List, Tuple, Optional
from ultralytics import YOLO


class YOLO26LP:
    """
    YOLO26-LP wrapper for DiL.
    Provides:
      1) Detect head raw output (pre-NMS)  -> Mo
      2) Post-NMS predicted bounding boxes -> GT 비교 / BL 계산
    """

    def __init__(
        self,
        weight_path: str,
        device: Optional[str] = None,
        conf_thresh: float = 0.5,
        iou_thresh: float = 0.5,
        img_size: int = 640,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.img_size = img_size

        # Load YOLO model
        self.model = YOLO(weight_path)
        self.model.to(self.device)
        self.model.eval()

        # storage for Detect head raw output (Mo)
        self._det_raw = None
        self._register_detect_hook()

    # ---------------------------------------------------------
    # Hook: Detect head (NMS 이전)
    # ---------------------------------------------------------
    def _register_detect_hook(self):
        """
        Register forward hook on Detect head.
        Ultralytics YOLO: Detect module is model.model[-1]
        """
        detect_layer = self.model.model.model[-1]

        def hook_fn(module, inputs, outputs):
            """
            outputs:
              - training: list of feature maps
              - inference: list / tuple depending on version
            """
            self._det_raw = outputs

        detect_layer.register_forward_hook(hook_fn)

    # ---------------------------------------------------------
    # Forward (used by DiL)
    # ---------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (1,3,H,W) float tensor in [0,1]

        Returns:
            det_raw   : Detect head raw output (pre-NMS)
            pred_boxes: List[List[float]]  (xyxy, pixel)
        """
        # --- Run inference (this triggers Detect hook) ---
        results = self.model.predict(
            source=x,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            imgsz=self.img_size,
            verbose=False,
        )

        # --- Post-NMS predictions ---
        r0 = results[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            pred_boxes = []
        else:
            pred_boxes = r0.boxes.xyxy.detach().cpu().tolist()

        # --- Detect head raw output (Mo) ---
        det_raw = self._det_raw
        if det_raw is None:
            raise RuntimeError(
                "Detect head raw output not captured. "
                "Check hook registration or Ultralytics version."
            )

        return det_raw, pred_boxes

    # ---------------------------------------------------------
    # Convenience: no-grad prediction only
    # ---------------------------------------------------------
    @torch.no_grad()
    def predict_boxes(self, x: torch.Tensor) -> List[List[float]]:
        """
        Only post-NMS boxes (for debugging / GT matching).
        """
        results = self.model.predict(
            source=x,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            imgsz=self.img_size,
            verbose=False,
        )
        r0 = results[0]
        if r0.boxes is None:
            return []
        return r0.boxes.xyxy.detach().cpu().tolist()
