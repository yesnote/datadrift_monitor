"""
YOLO class-wise validation using Ultralytics official model.val()

- Input:
  --root : runs/<exp_name> (expects weights/best.pt)
  --data : dataset.yaml path
- Output:
  - results saved to:
    runs/<exp_name>/detect/val
  - console: class-wise AP, overall mAP
"""

from __future__ import annotations

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        required=True,
        help="run root directory (e.g., runs/yolo26-LP)",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="dataset.yaml path for validation",
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()

    root = Path(args.root)
    model_path = root / "weights" / "best.pt"
    assert model_path.exists(), f"model not found: {model_path}"

    model = YOLO(str(model_path))

    metrics = model.val(
        data=args.data,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        split="val",
        project=str(root),
        name="val",
        verbose=True,
    )

    # -------------------------
    # Class-wise summary
    # -------------------------
    print("\n=== Class-wise AP (mAP50) ===")
    for i, name in metrics.names.items():
        print(f"{name:10s}: {metrics.box.maps[i]:.4f}")

    print("\n=== Overall ===")
    print(f"mAP50     : {metrics.box.map50:.4f}")
    print(f"mAP50-95  : {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()

# python val_yolo26_lp.py --root runs/yolo26-LP --data data/dataset_lp_balanced_val.yaml
