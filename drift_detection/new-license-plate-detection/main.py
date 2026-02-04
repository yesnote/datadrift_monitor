import yaml
import torch
import numpy as np

from data.dataloader import build_dataloader
from eval.missed_detection import is_missed_detection
from utils.metrics import compute_binary_classification_metrics
from dil.dil_score import compute_dil
from xai.gradcam_yolo import YOLOGradCAM
from model.yolo26_lp import YOLO26LP

# 1. load config
with open("config/dataset_lp.yaml") as f:
    cfg = yaml.safe_load(f)

# 2. load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolo26_lp_model = torch.load(cfg["model"]["weight"], map_location=device)
yolo26_lp_model.eval().to(device)

detector = YOLO26LP(yolo26_lp_model)

# 3. dataloader
loader = build_dataloader(cfg)

# 4. GradCAM (target layer는 backbone/neck의 conv layer 1개를 쓰는 것이 권장)
gradcam = YOLOGradCAM(
    yolo26_lp_model,
    target_layer=yolo26_lp_model.model[24].m[0]  # 예시: 적절한 conv layer로 조정
)

# 5. inference + DiL
dils = []
missed_flags = []

for batch in loader:
    x = batch["image"].to(device)
    gt_boxes = batch["gt_boxes"]

    det_raw, pred_boxes = detector.forward(x)
    cam = gradcam.saliency(det_raw, input_shape=x.shape)

    dil = compute_dil(cam, pred_boxes)
    missed = is_missed_detection(
        gt_boxes,
        pred_boxes,
        iou_thresh=cfg["model"]["iou_thresh"]
    )

    dils.append(dil)
    missed_flags.append(int(missed))

# 6. PR / ROC metrics (sklearn)
metrics = compute_binary_classification_metrics(
    scores=dils,
    labels=missed_flags,
    compute_roc=True,
)

print(f"AP (Missed Detection): {metrics['pr']['ap']:.4f}")
if "roc" in metrics:
    print(f"ROC AUC: {metrics['roc']['auc']:.4f}")
