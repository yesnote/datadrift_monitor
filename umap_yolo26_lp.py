from __future__ import annotations

import argparse
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import umap
from ultralytics import YOLO
from tqdm import tqdm
import cv2


# -------------------------
# Args
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="YOLO weight path (.pt)")
    parser.add_argument("--data", required=True, help="val image directory")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--out", default="umap_layer22.png")
    return parser.parse_args()


# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()

    device = f"cuda:{args.device}" if args.device != "cpu" else "cpu"

    model = YOLO(args.model)
    net = model.model.to(device).eval()

    features = []

    # hook: layer 22
    def hook_fn(module, inp, out):
        # out: (B, C, H, W)
        f = out.mean(dim=(2, 3))  # GAP
        features.append(f.detach().cpu())

    handle = net.model[22].register_forward_hook(hook_fn)

    img_dir = Path(args.data)
    imgs = sorted(list(img_dir.glob("*.jpg")))

    with torch.no_grad():
        for img_path in tqdm(imgs, desc="Extract features"):
            img = cv2.imread(str(img_path))
            img = cv2.resize(img, (args.imgsz, args.imgsz))
            img = img[:, :, ::-1]  # BGR → RGB
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            img = img.unsqueeze(0).to(device)

            _ = net(img)

    handle.remove()

    X = torch.cat(features, dim=0).numpy()  # (N, C)

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

    plt.figure(figsize=(7, 7))
    plt.scatter(X_umap[:, 0], X_umap[:, 1], s=5)
    plt.title("YOLO26 Layer22 Feature UMAP")
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    plt.close()

    print(f"[DONE] saved: {args.out}")


if __name__ == "__main__":
    main()
