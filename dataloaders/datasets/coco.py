import json
import os
from collections import defaultdict

import torch
from torch.utils.data import Dataset

from dataloaders.utils.data_utils import list_image_files, read_image_as_rgb


class COCODataset(Dataset):
    """
    Expected COCO layout:
    - <root>/images/<split>/*.jpg
    - <root>/annotations/instances_<split>.json
    """

    def __init__(self, root, split="train2017", image_dir=None, annotation_file=None, img_size=640):
        self.root = root
        self.split = split
        self.img_size = img_size

        self.image_dir = image_dir or os.path.join(root, "images", split)
        self.annotation_file = annotation_file or os.path.join(
            root, "annotations", f"instances_{split}.json"
        )

        self.images = list_image_files(self.image_dir)
        self.image_id_by_name = {}
        self.annotations_by_image_id = defaultdict(list)

        if os.path.isfile(self.annotation_file):
            self._load_annotations()

    def _load_annotations(self):
        with open(self.annotation_file, "r", encoding="utf-8") as f:
            payload = json.load(f)

        for image in payload.get("images", []):
            self.image_id_by_name[image["file_name"]] = image["id"]

        for ann in payload.get("annotations", []):
            self.annotations_by_image_id[ann["image_id"]].append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        image = read_image_as_rgb(image_path)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        file_name = os.path.basename(image_path)
        image_id = self.image_id_by_name.get(file_name, index)
        anns = self.annotations_by_image_id.get(image_id, [])

        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "path": image_path,
        }
        return image, target
