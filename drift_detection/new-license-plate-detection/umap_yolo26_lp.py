"""
YOLO26n model's 22nd layer feature visualization using UMAP.
22nd layer: semantic feature right before detection head.

- Load model from <root>/weights/best.pt
- Extract layer-22 features
- Save:
  - features + labels -> <root>/umap/umap_layer22.npz
  - UMAP image       -> <root>/umap/umap_layer22.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import umap
from ultralytics import YOLO
from tqdm import tqdm


# -------------------------
# Args
# -------------------------
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
        help="dataset_lp_balanced_val root",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
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

    out_dir = root / "umap"
    out_dir.mkdir(parents=True, exist_ok=True)

    png_path = out_dir / "umap_layer22.png"
    npz_path = out_dir / "umap_layer22.npz"

    device = f"cuda:{args.device}" if args.device != "cpu" else "cpu"

    # -------------------------
    # Load model
    # -------------------------
    model = YOLO(str(model_path))
    net = model.model.to(device).eval()

    img_dir = Path(args.data) / "images" / "val"
    meta_dir = Path(args.data) / "meta" / "val"

    features = []
    labels = []

    # hook: layer 22
    def hook_fn(module, inp, out):
        f = out.mean(dim=(2, 3))  # GAP
        features.append(f.detach().cpu())

    handle = net.model[22].register_forward_hook(hook_fn)

    img_paths = sorted(img_dir.glob("*.jpg"))

    # -------------------------
    # Feature extraction
    # -------------------------
    with torch.no_grad():
        for img_path in tqdm(img_paths, desc="Extract features"):
            stem = img_path.stem
            meta_path = meta_dir / f"{stem}.json"
            if not meta_path.exists():
                continue

            class_id = read_class_id(meta_path)

            img = cv2.imread(str(img_path))
            img = cv2.resize(img, (args.imgsz, args.imgsz))
            img = img[:, :, ::-1].copy()  # BGR -> RGB (avoid negative stride)
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            img = img.unsqueeze(0).to(device)

            _ = net(img)
            labels.append(class_id)

    handle.remove()

    X = torch.cat(features, dim=0).numpy()  # (N, C)
    y = np.array(labels)

    # -------------------------
    # Save features
    # -------------------------
    np.savez(
        npz_path,
        features=X,
        labels=y,
    )
    print(f"[SAVE] features -> {npz_path}")

    # -------------------------
    # UMAP
    # -------------------------
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        random_state=42,
    )
    X_umap = reducer.fit_transform(X)

    # -------------------------
    # Plot (PNG)
    # -------------------------
    plt.figure(figsize=(7, 7))
    classes = sorted(set(y))
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))

    for cls, c in zip(classes, colors):
        mask = y == cls
        plt.scatter(
            X_umap[mask, 0],
            X_umap[mask, 1],
            s=8,
            color=c,
            label=cls,
        )

    plt.legend(markerscale=2)
    plt.title("YOLO26 Layer22 UMAP (by class_ID)")
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()

    print(f"[SAVE] UMAP image -> {png_path}")
    print("[DONE]")


if __name__ == "__main__":
    main()

# python umap_yolo26_lp.py --root runs/yolo26-LP --data dataset_lp_balanced_val