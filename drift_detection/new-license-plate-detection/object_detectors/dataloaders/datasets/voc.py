import os
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset

from dataloaders.utils.data_utils import list_image_files, pascal_voc_names, read_image_as_rgb


class VOCDataset(Dataset):
    """
    Expected VOC layout:
    - <root>/JPEGImages/*.jpg
    - <root>/Annotations/*.xml
    - <root>/ImageSets/Main/<split>.txt
    """

    def __init__(self, root, split="train", img_size=640):
        self.root = root
        self.split = split
        self.img_size = img_size

        self.image_dir = os.path.join(root, "JPEGImages")
        self.annotation_dir = os.path.join(root, "Annotations")
        self.split_file = os.path.join(root, "ImageSets", "Main", f"{split}.txt")
        self.class_to_idx = {name: idx for idx, name in enumerate(pascal_voc_names)}
        self.images = self._resolve_images()

    def _resolve_images(self):
        if os.path.isfile(self.split_file):
            image_paths = []
            with open(self.split_file, "r", encoding="utf-8") as f:
                for line in f:
                    stem = line.strip().split()[0]
                    image_paths.append(os.path.join(self.image_dir, f"{stem}.jpg"))
            return [p for p in image_paths if os.path.isfile(p)]
        return list_image_files(self.image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        image = read_image_as_rgb(image_path)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        ann_path = os.path.join(
            self.annotation_dir,
            f"{os.path.splitext(os.path.basename(image_path))[0]}.xml",
        )
        boxes = []
        labels = []
        gt_class_names = []
        if os.path.isfile(ann_path):
            root = ET.parse(ann_path).getroot()
            for obj in root.findall("object"):
                label_name = obj.findtext("name", default="__background__")
                bbox = obj.find("bndbox")
                if bbox is None:
                    continue
                xmin = float(bbox.findtext("xmin", default="0"))
                ymin = float(bbox.findtext("ymin", default="0"))
                xmax = float(bbox.findtext("xmax", default="0"))
                ymax = float(bbox.findtext("ymax", default="0"))
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(self.class_to_idx.get(label_name, 0))
                gt_class_names.append(label_name)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([index], dtype=torch.int64),
            "path": image_path,
            "dataset_name": "voc",
            "gt_class_names": gt_class_names,
        }
        return image, target
