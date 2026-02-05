import yaml
import torch
import numpy as np
import os
import csv
import cv2

from data.dataloader import build_dataloader
from eval.missed_detection import is_missed_detection
from utils.metrics import compute_binary_classification_metrics
from dil.dil_score import compute_dil
from xai.gradcam_yolo import YOLOGradCAM
from model.yolo26_lp import YOLO26LP


# 1. load config
with open("config/yolo26-LP.yaml") as f:
    cfg = yaml.safe_load(f)

os.makedirs(cfg["output"]["vis_dir"], exist_ok=True)

# 2. load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolo26_lp_model = torch.load(cfg["model"]["weight"], map_location=device)
yolo26_lp_model.eval().to(device)

detector = YOLO26LP(yolo26_lp_model)

# 3. dataloader
loader = build_dataloader(cfg)

# 4. GradCAM
layer_idx = cfg["xai"]["layer"]
target_layer = yolo26_lp_model.model.model[layer_idx]

gradcam = YOLOGradCAM(
    yolo26_lp_model,
    target_layer=target_layer
)

# 5. inference + DiL
dils = []
missed_flags = []

csv_rows = []

for idx, batch in enumerate(loader):
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

    # save csv row
    csv_rows.append({
        "index": idx,
        "dil": float(dil),
        "missed": int(missed),
    })

    # visualization
    img = x[0].detach().cpu().permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)

    heatmap = cam[0, 0].detach().cpu().numpy()
    heatmap = cv2.applyColorMap(
        np.uint8(255 * heatmap),
        cv2.COLORMAP_JET
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

    overlay = np.clip(img + heatmap * 0.5, 0, 1)
    overlay = (overlay * 255).astype(np.uint8)

    cv2.imwrite(
        os.path.join(cfg["output"]["vis_dir"], f"{idx:05d}.jpg"),
        overlay[:, :, ::-1],  # RGB → BGR
    )

# 6. save DiL CSV
with open(cfg["output"]["dil_csv"], "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["index", "dil", "missed"])
    writer.writeheader()
    writer.writerows(csv_rows)

# 7. PR / ROC metrics
metrics = compute_binary_classification_metrics(
    scores=dils,
    labels=missed_flags,
    compute_roc=True,
)

print(f"AP (Missed Detection): {metrics['pr']['ap']:.4f}")
if "roc" in metrics:
    print(f"ROC AUC: {metrics['roc']['auc']:.4f}")
