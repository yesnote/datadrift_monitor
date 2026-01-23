"""Fine-tune a pretrained YOLO26 model on a custom dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a YOLO26 pretrained model on a custom dataset."
    )
    parser.add_argument(
        "--model",
        default="yolo_LP_tuning/yolo26x.pt",
        help="Path or name of the pretrained YOLO26 checkpoint.",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to the dataset YAML file (Ultralytics format).",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size.")
    parser.add_argument("--batch", type=int, default=1, help="Batch size.")
    parser.add_argument(
        "--device",
        default="",
        help="Device to use (e.g., '0', '0,1', 'cpu').",
    )
    parser.add_argument(
        "--project",
        default="yolo_LP_tuning/runs/train",
        help="Project directory for training outputs.",
    )
    parser.add_argument(
        "--name",
        default="yolo26-LP",
        help="Run name for training outputs.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {data_path}. "
            "Update --data to point to your dataset config."
        )

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        workers=args.workers,
        resume=args.resume,
        cos_lr=True,
        freeze=5,
    )


if __name__ == "__main__":
    main()

# python train_yolo26.py --data custom_dataset.yaml --device 0 --workers 4