from pathlib import Path

from torch.utils.data import DataLoader
import yaml

from dataloaders.datasets.coco import COCODataset
from dataloaders.datasets.openimages import OpenImagesDataset
from dataloaders.datasets.voc import VOCDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_mode(config):
    mode = str(config.get("mode", "train")).lower()
    valid_modes = {"train", "test", "predict"}
    if mode not in valid_modes:
        raise ValueError(f"Unsupported mode: {mode}. Expected one of {sorted(valid_modes)}")
    return mode


def get_active_dataset_cfg(config):
    dataset_cfg = config["dataset"]
    used_dataset = str(dataset_cfg["used_dataset"]).lower()
    if used_dataset not in dataset_cfg:
        raise ValueError(
            f"dataset.used_dataset='{used_dataset}' but dataset.{used_dataset} is not defined."
        )
    return used_dataset, dataset_cfg[used_dataset]


def build_dataset(config, split="train"):
    name, dataset_cfg = get_active_dataset_cfg(config)
    root_path = Path(dataset_cfg["root"])
    if not root_path.is_absolute():
        root_path = (PROJECT_ROOT / root_path).resolve()
    root = str(root_path)

    if name == "coco":
        split_key = f"{split}_split"
        coco_split = dataset_cfg.get(split_key, split)
        ann_file_key = f"{split}_annotation_file"
        ann_file = dataset_cfg.get(ann_file_key)
        ann_path = None
        if ann_file:
            ann_path = str(Path(root) / dataset_cfg["annotation_dir"] / ann_file)
        image_dir = str(Path(root) / dataset_cfg["image_dir"] / coco_split)
        if not Path(image_dir).is_dir():
            fallback_dir = Path(root) / coco_split
            if fallback_dir.is_dir():
                image_dir = str(fallback_dir)
        return COCODataset(
            root=root,
            split=coco_split,
            image_dir=image_dir,
            annotation_file=ann_path,
            img_size=config["model"]["img_size"],
        )

    if name in {"voc", "pascal_voc"}:
        split_key = f"{split}_split"
        voc_split = dataset_cfg.get(split_key, split)
        return VOCDataset(root=root, split=voc_split, img_size=config["model"]["img_size"])

    if name in {"openimages", "open_images", "oid"}:
        split_key = f"{split}_split"
        oi_split = dataset_cfg.get(split_key, split)
        class_names = dataset_cfg.get("class_names")
        return OpenImagesDataset(
            root=root,
            split=oi_split,
            class_names=class_names,
            img_size=config["model"]["img_size"],
        )

    raise ValueError(f"Unsupported dataset name: {name}")


def yolo_collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def create_dataloader(config, split="train"):
    _ = get_mode(config)
    dataset = build_dataset(config, split=split)
    dl_cfg = config["dataloader"]
    shuffle = dl_cfg["shuffle_train"] if split == "train" else dl_cfg["shuffle_eval"]
    return DataLoader(
        dataset,
        batch_size=dl_cfg["batch_size"],
        shuffle=shuffle,
        num_workers=dl_cfg["num_workers"],
        pin_memory=dl_cfg["pin_memory"],
        collate_fn=yolo_collate_fn,
    )
