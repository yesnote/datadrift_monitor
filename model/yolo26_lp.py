class YOLO26LP:
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device

    def forward(self, x):
        """
        Returns:
          det_raw   : Detect head raw outputs (pre-NMS)
          pred_boxes: list of (x1,y1,x2,y2)
        """
        det_raw, preds = self.model(x)
        pred_boxes = preds["boxes"]  # 구현체에 맞게 수정
        return det_raw, pred_boxes
