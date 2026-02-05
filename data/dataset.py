import os
import torch
import numpy as np
from PIL import Image

class LPDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, split="val"):
        self.root = cfg["dataset"]["root"]
        self.img_dir = os.path.join(self.root, cfg["dataset"]["images"][split])
        self.label_dir = os.path.join(self.root, cfg["dataset"]["labels"][split])
        self.img_size = tuple(cfg["dataloader"]["img_size"])

        self.images = sorted(os.listdir(self.img_dir))

    def _load_label(self, label_path, w, h):
        boxes = []
        if not os.path.exists(label_path):
            return boxes

        with open(label_path, "r") as f:
            for line in f:
                _, cx, cy, bw, bh = map(float, line.split())
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                x2 = (cx + bw / 2) * w
                y2 = (cy + bh / 2) * h
                boxes.append([x1, y1, x2, y2])
        return torch.tensor(boxes)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace(".jpg", ".txt"))

        img = Image.open(img_path).convert("RGB").resize(self.img_size)
        w, h = img.size
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

        gt_boxes = self._load_label(label_path, w, h)

        return {
            "image": img.unsqueeze(0),
            "gt_boxes": gt_boxes,
            "img_id": img_name
        }

    def __len__(self):
        return len(self.images)
