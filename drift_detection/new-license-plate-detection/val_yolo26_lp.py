from __future__ import annotations

import argparse
from pathlib import Path
import json
from collections import defaultdict

import torch
from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou
from torch.utils.tensorboard import SummaryWriter


# -------------------------
# Args
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="runs/yolo26-LP")
    parser.add_argument("--data", required=True, help="validation dataset root")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    return parser.parse_args()


# -------------------------
# Utils
# -------------------------
def read_class_id(meta_path: Path) -> str:
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    anns = meta["Learning_Data_Info"]["annotations"]
    return anns[0]["license_plate"][0]["class_ID"]


# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()

    root = Path(args.root)
    model_path = root / "weights" / "best.pt"
    assert model_path.exists(), f"model not found: {model_path}"

    device = f"cuda:{args.device}" if args.device != "cpu" else "cpu"

    # TensorBoard
    tb_dir = root / "val_classwise"
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(tb_dir)

    model = YOLO(str(model_path))

    # class-wise stats
    stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    # -------------------------
    # Validation loop (custom)
    # -------------------------
    results = model.predict(
        source=str(Path(args.data) / "images" / "val"),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=device,
        stream=True,
        save=False,
        verbose=False,
    )

    img_dir = Path(args.data) / "images" / "val"
    meta_dir = Path(args.data) / "meta" / "val"
    label_dir = Path(args.data) / "labels" / "val"

    for r in results:
        stem = Path(r.path).stem
        meta_path = meta_dir / f"{stem}.json"
        label_path = label_dir / f"{stem}.txt"

        if not meta_path.exists() or not label_path.exists():
            continue

        class_id = read_class_id(meta_path)

        # GT boxes
        gt = []
        with open(label_path) as f:
            for line in f:
                _, x, y, w, h = map(float, line.split())
                gt.append([x, y, w, h])
        gt = torch.tensor(gt, device=device) if gt else torch.empty((0, 4), device=device)

        # Pred boxes
        pred = r.boxes.xywhn if r.boxes is not None else torch.empty((0, 4), device=device)

        if len(gt) == 0 and len(pred) == 0:
            continue

        if len(gt) == 0:
            stats[class_id]["fp"] += len(pred)
            continue

        if len(pred) == 0:
            stats[class_id]["fn"] += len(gt)
            continue

        ious = box_iou(
            r.boxes.xyxy,
            r.boxes.xyxy.new_tensor(r.boxes.xyxy),
        )

        matched_gt = set()
        matched_pred = set()

        for i in range(len(pred)):
            max_iou, j = ious[i].max(0)
            if max_iou >= args.iou:
                matched_pred.add(i)
                matched_gt.add(j.item())

        stats[class_id]["tp"] += len(matched_pred)
        stats[class_id]["fp"] += len(pred) - len(matched_pred)
        stats[class_id]["fn"] += len(gt) - len(matched_gt)

    # -------------------------
    # Report + TensorBoard
    # -------------------------
    print("\n=== Class-wise Validation Result ===")
    for cid, s in stats.items():
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)

        print(f"{cid:6s} | TP {tp:5d} FP {fp:5d} FN {fn:5d} | "
              f"P {prec:.4f} R {rec:.4f}")

        writer.add_scalar(f"{cid}/precision", prec, 0)
        writer.add_scalar(f"{cid}/recall", rec, 0)

    writer.close()
    print(f"\n[TENSORBOARD] {tb_dir}")
    print("[DONE]")


if __name__ == "__main__":
    main()

# python val_yolo26_lp.py --root runs/yolo26-LP --data dataset_lp_balanced_val