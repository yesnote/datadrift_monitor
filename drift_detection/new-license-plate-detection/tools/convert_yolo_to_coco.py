import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert YOLO txt labels to COCO JSON + objects_count.csv."
    )
    parser.add_argument(
        "--dataset-dir",
        help="Dataset root path that contains 'images' and 'labels' folders.",
    )
    parser.add_argument("--images-dir", help="Path to images directory.")
    parser.add_argument(
        "--labels-dir", help="Path to YOLO labels directory (txt files)."
    )
    parser.add_argument("--output-dir", help="Path to output annotations directory.")
    parser.add_argument(
        "--split",
        help="Optional split name under images/ and labels/ (e.g. val/train/test).",
    )
    parser.add_argument(
        "--ext", default="jpg,jpeg,png", help="Comma-separated image extensions."
    )
    parser.add_argument(
        "--class-names", default="license_plate", help="Comma-separated class names."
    )
    return parser.parse_args()


def list_images(images_dir: str, exts: List[str]) -> List[Path]:
    images_dir_path = Path(images_dir)
    exts = [e.lower().lstrip(".") for e in exts]
    images = []
    for ext in exts:
        images.extend(images_dir_path.glob(f"*.{ext}"))
        images.extend(images_dir_path.glob(f"*.{ext.upper()}"))
    # Deduplicate on case-insensitive file systems (e.g., Windows)
    unique = {str(p).lower(): p for p in images}
    return sorted(unique.values())


def yolo_to_coco_bbox(
    xywh_norm: Tuple[float, float, float, float], width: int, height: int
) -> Tuple[float, float, float, float]:
    x_center, y_center, w_norm, h_norm = xywh_norm
    w = w_norm * width
    h = h_norm * height
    x_min = (x_center * width) - (w / 2.0)
    y_min = (y_center * height) - (h / 2.0)
    return x_min, y_min, w, h


def resolve_input_dirs(
    dataset_dir: Path, exts: List[str], split: str = None
) -> Tuple[Path, Path]:
    images_root = dataset_dir / "images"
    labels_root = dataset_dir / "labels"
    if not images_root.exists() or not labels_root.exists():
        raise SystemExit(
            f"'images' and 'labels' folders are required under dataset dir: {dataset_dir}"
        )

    if split:
        images_dir = images_root / split
    else:
        images_dir = images_root
        if not list_images(str(images_dir), exts):
            split_candidates = [
                name for name in ["val", "train", "test"] if (images_root / name).exists()
            ]
            split_with_images = [
                name
                for name in split_candidates
                if list_images(str(images_root / name), exts)
            ]
            if not split_with_images:
                raise SystemExit(
                    f"No images found in {images_root} (or val/train/test subfolders)."
                )
            # Prefer val when multiple splits exist.
            selected = "val" if "val" in split_with_images else split_with_images[0]
            images_dir = images_root / selected
            split = selected

    if not images_dir.exists():
        raise SystemExit(f"Images directory does not exist: {images_dir}")

    labels_dir = labels_root / split if split and (labels_root / split).exists() else labels_root
    return images_dir, labels_dir


def main():
    args = parse_args()
    exts = [e.strip() for e in args.ext.split(",") if e.strip()]
    class_names = [c.strip() for c in args.class_names.split(",") if c.strip()]
    dataset_mode = bool(args.dataset_dir)

    if dataset_mode:
        dataset_dir = Path(args.dataset_dir)
        images_dir, labels_dir = resolve_input_dirs(dataset_dir, exts, args.split)
        annotations_path = dataset_dir / "annotations.json"
        objects_count_path = dataset_dir / "objects_count.csv"
    else:
        if not (args.images_dir and args.labels_dir and args.output_dir):
            raise SystemExit(
                "Either provide --dataset-dir, or provide --images-dir/--labels-dir/--output-dir."
            )
        images_dir = Path(args.images_dir)
        labels_dir = Path(args.labels_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset_root = output_dir.parent
        objects_count_dir = dataset_root / "objects_count"
        objects_count_dir.mkdir(parents=True, exist_ok=True)
        annotations_path = output_dir / "annotations.json"
        objects_count_path = objects_count_dir / "objects_count.csv"

    images = list_images(str(images_dir), exts)
    if not images:
        raise SystemExit(f"No images found in {images_dir} with extensions: {exts}")

    coco = {
        "info": {"description": "YOLO to COCO conversion"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name} for i, name in enumerate(class_names)],
    }

    ann_id = 1
    objects_count_rows = [("image_id", "count")]

    for image_id, img_path in enumerate(images):
        with Image.open(img_path) as im:
            width, height = im.size

        file_stem = img_path.stem
        coco["images"].append(
            {
                "id": image_id,
                "file_name": img_path.name,
                "width": width,
                "height": height,
            }
        ),

        label_path = labels_dir / f"{file_stem}.txt"
        count = 0
        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    class_id = int(float(parts[0]))
                    x, y, w, h = map(float, parts[1:5])
                    bbox = yolo_to_coco_bbox((x, y, w, h), width, height)
                    coco["annotations"].append(
                        {
                            "id": ann_id,
                            "image_id": image_id,
                            "category_id": class_id,
                            "bbox": [float(v) for v in bbox],
                            "area": float(bbox[2] * bbox[3]),
                            "iscrowd": 0,
                            "original_image_id": file_stem,
                        }
                    )
                    ann_id += 1
                    count += 1

        objects_count_rows.append((file_stem, str(count)))

    with open(annotations_path, "w", encoding="utf-8") as f:
        json.dump(coco, f)

    with open(objects_count_path, "w", encoding="utf-8") as f:
        for row in objects_count_rows:
            f.write(",".join(row) + "\n")

    print(f"Wrote COCO annotations: {annotations_path}")
    print(f"Wrote objects count CSV: {objects_count_path}")


if __name__ == "__main__":
    main()

# python tools/convert_yolo_to_coco.py --dataset-dir dataset/dataset_lp_small --ext "jpg" --class-names "license_plate"
