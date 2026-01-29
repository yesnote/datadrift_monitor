"""
YOLO26n model's 22nd layer feature visualization using UMAP.

- Extract features from layer 22 (pre-detection semantic feature)
- Save features to .npz
- Reload saved features if exists
- Visualize UMAP:
  - static PNG
  - interactive HTML (browser)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import cv2
import torch
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import plotly.express as px
from ultralytics import YOLO
from tqdm import tqdm


# -------------------------
# Args
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="YOLO weight (.pt)")
    parser.add_argument("--data", required=True, help="dataset_lp_balanced_val root")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--out", default="umap_layer22")
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Reuse saved feature file if exists",
    )
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
# Feature Extraction
# -------------------------
def extract_features(
    model_path: Path,
    data_root: Path,
    imgsz: int,
    device: str,
):
    model = YOLO(str(model_path))
    net = model.model.to(device).eval()

    img_dir = data_root / "images" / "val"
    meta_dir = data_root / "meta" / "val"

    features = []
    labels = []

    # hook: layer 22
    def hook_fn(module, inp, out):
        f = out.mean(dim=(2, 3))  # GAP
        features.append(f.detach().cpu())

    handle = net.model[22].register_forward_hook(hook_fn)

    img_paths = sorted(img_dir.glob("*.jpg"))

    with torch.no_grad():
        for img_path in tqdm(img_paths, desc="Extract features"):
            stem = img_path.stem
            meta_path = meta_dir / f"{stem}.json"
            if not meta_path.exists():
                continue

            class_id = read_class_id(meta_path)

            img = cv2.imread(str(img_path))
            img = cv2.resize(img, (imgsz, imgsz))
            img = img[:, :, ::-1].copy()  # BGR → RGB, avoid negative stride
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            img = img.unsqueeze(0).to(device)

            _ = net(img)
            labels.append(class_id)

    handle.remove()

    X = torch.cat(features, dim=0).numpy()  # (N, C)
    y = np.array(labels)

    return X, y


# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()
    device = f"cuda:{args.device}" if args.device != "cpu" else "cpu"

    out_base = Path(args.out)
    feat_path = out_base.with_suffix(".npz")
    png_path = out_base.with_suffix(".png")
    html_path = out_base.with_suffix(".html")

    # -------------------------
    # Load or Extract Features
    # -------------------------
    if args.reuse and feat_path.exists():
        print(f"[LOAD] loading features from {feat_path}")
        data = np.load(feat_path, allow_pickle=True)
        X = data["features"]
        y = data["labels"]
    else:
        print("[INFO] extracting features from model")
        X, y = extract_features(
            model_path=Path(args.model),
            data_root=Path(args.data),
            imgsz=args.imgsz,
            device=device,
        )

        np.savez(
            feat_path,
            features=X,
            labels=y,
        )
        print(f"[SAVE] features saved to {feat_path}")

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
    # Static plot (PNG)
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

    print(f"[SAVE] static UMAP saved to {png_path}")

    # -------------------------
    # Interactive plot (HTML)
    # -------------------------
    df = pd.DataFrame({
        "x": X_umap[:, 0],
        "y": X_umap[:, 1],
        "class_ID": y,
    })

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="class_ID",
        title="YOLO26 Layer22 UMAP (by class_ID)",
        opacity=0.85,
    )

    fig.write_html(html_path)
    print(f"[SAVE] interactive UMAP saved to {html_path}")
    print("[DONE]")


if __name__ == "__main__":
    main()
