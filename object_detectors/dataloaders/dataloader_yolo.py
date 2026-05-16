import hashlib
import math
from pathlib import Path

from torch.utils.data import ConcatDataset, DataLoader, Subset
import yaml

from dataloaders.datasets.coco import COCODataset
from dataloaders.datasets.openimages import OpenImagesDataset
from dataloaders.datasets.road import (
    BDD100KDataset,
    CityscapesDetectionDataset,
    FoggyCityscapesDetectionDataset,
    KITTIDataset,
)
from dataloaders.datasets.voc import VOCDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_dataset_path(root, value):
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str(Path(root) / path)


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_mode(config):
    mode = str(config.get("mode", "train")).lower()
    valid_modes = {"train", "test", "predict"}
    if mode not in valid_modes:
        raise ValueError(f"Unsupported mode: {mode}. Expected one of {sorted(valid_modes)}")
    return mode


def _normalize_dataset_names(dataset_cfg):
    raw = dataset_cfg["used_dataset"]
    if isinstance(raw, str):
        names = [raw.strip().lower()]
    elif isinstance(raw, (list, tuple)):
        names = [str(v).strip().lower() for v in raw if str(v).strip()]
    else:
        raise ValueError("dataset.used_dataset must be a string or list of strings.")
    if not names:
        raise ValueError("dataset.used_dataset is empty.")
    return names


def _build_single_dataset(name, dataset_cfg, root, split_key, img_size):
    if name == "coco":
        coco_split = dataset_cfg.get(f"{split_key}_split", split_key)
        ann_file_key = f"{split_key}_annotation_file"
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
            img_size=img_size,
        )

    if name in {"voc", "pascal_voc"}:
        voc_split = dataset_cfg.get(f"{split_key}_split", split_key)
        return VOCDataset(root=root, split=voc_split, img_size=img_size)

    if name in {"openimages", "open_images", "oid"}:
        oi_split = dataset_cfg.get(f"{split_key}_split", split_key)
        min_gt_boxes = int(dataset_cfg.get("min_gt_boxes", 0))
        return OpenImagesDataset(
            root=root,
            split=oi_split,
            img_size=img_size,
            min_gt_boxes=min_gt_boxes,
        )

    if name == "kitti":
        kitti_split = dataset_cfg.get(f"{split_key}_split", split_key)
        split_file = dataset_cfg.get(f"{split_key}_split_file") or dataset_cfg.get("split_file")
        return KITTIDataset(
            root=root,
            split=kitti_split,
            img_size=img_size,
            image_dir=_resolve_dataset_path(root, dataset_cfg.get("image_dir")),
            label_dir=_resolve_dataset_path(root, dataset_cfg.get("label_dir")),
            split_file=_resolve_dataset_path(root, split_file),
        )

    if name in {"bdd100k", "bdd"}:
        bdd_split = dataset_cfg.get(f"{split_key}_split", split_key)
        ann_file = dataset_cfg.get(f"{split_key}_annotation_file") or dataset_cfg.get("annotation_file")
        image_dir = dataset_cfg.get(f"{split_key}_image_dir") or dataset_cfg.get("image_dir")
        return BDD100KDataset(
            root=root,
            split=bdd_split,
            img_size=img_size,
            image_dir=_resolve_dataset_path(root, image_dir) if image_dir else None,
            annotation_file=_resolve_dataset_path(root, ann_file) if ann_file else None,
        )

    if name == "cityscapes":
        city_split = dataset_cfg.get(f"{split_key}_split", split_key)
        image_dir = dataset_cfg.get(f"{split_key}_image_dir") or dataset_cfg.get("image_dir")
        annotation_dir = dataset_cfg.get(f"{split_key}_annotation_dir") or dataset_cfg.get("annotation_dir")
        return CityscapesDetectionDataset(
            root=root,
            split=city_split,
            img_size=img_size,
            image_dir=_resolve_dataset_path(root, image_dir) if image_dir else None,
            annotation_dir=_resolve_dataset_path(root, annotation_dir) if annotation_dir else None,
        )

    if name in {"foggy_cityscapes", "foggy_city"}:
        foggy_split = dataset_cfg.get(f"{split_key}_split", split_key)
        image_dir = dataset_cfg.get(f"{split_key}_image_dir") or dataset_cfg.get("image_dir")
        annotation_dir = dataset_cfg.get(f"{split_key}_annotation_dir") or dataset_cfg.get("annotation_dir")
        return FoggyCityscapesDetectionDataset(
            root=root,
            split=foggy_split,
            img_size=img_size,
            image_dir=_resolve_dataset_path(root, image_dir) if image_dir else None,
            annotation_dir=_resolve_dataset_path(root, annotation_dir) if annotation_dir else None,
        )

    raise ValueError(f"Unsupported dataset name: {name}")


def _sample_key(dataset, index, dataset_name, split_key):
    if hasattr(dataset, "images") and index < len(getattr(dataset, "images")):
        sample_id = str(dataset.images[index])
    elif hasattr(dataset, "samples") and index < len(getattr(dataset, "samples")):
        sample = dataset.samples[index]
        sample_id = str(sample.get("image_path", index)) if isinstance(sample, dict) else str(index)
    else:
        sample_id = str(index)
    return f"{dataset_name}|{split_key}|{sample_id}"


def _apply_used_ratio(dataset, dataset_cfg, dataset_name, split_key):
    ratio = float(dataset_cfg.get("used_ratio", 1.0))
    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"dataset.used_ratio for '{dataset_name}' must be in (0, 1].")
    n = len(dataset)
    if ratio >= 1.0 or n == 0:
        return dataset

    keep = max(1, int(math.ceil(n * ratio)))
    ranked = []
    for idx in range(n):
        key = _sample_key(dataset, idx, dataset_name, split_key)
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        ranked.append((digest, idx))
    selected = sorted(idx for _digest, idx in sorted(ranked)[:keep])
    return Subset(dataset, selected)


def build_dataset(config, split="train"):
    mode = get_mode(config)
    root_dataset_cfg = config["dataset"]
    names = _normalize_dataset_names(root_dataset_cfg)
    used_ratios = root_dataset_cfg.get("used_ratio", 1.0)

    if isinstance(split, (list, tuple)):
        split_keys = [str(v).strip() for v in split if str(v).strip()]
    else:
        split_keys = [str(split).strip()] * len(names)

    if len(split_keys) != len(names):
        raise ValueError(
            f"Length mismatch: used_dataset has {len(names)} entries but split has {len(split_keys)} entries."
        )
    if isinstance(used_ratios, (list, tuple)):
        ratio_values = [float(v) for v in used_ratios]
    else:
        ratio_values = [float(used_ratios)] * len(names)
    if len(ratio_values) != len(names):
        raise ValueError(
            f"Length mismatch: used_dataset has {len(names)} entries but dataset.used_ratio has {len(ratio_values)} entries."
        )

    datasets = []
    img_size = config["model"]["img_size"]
    for name, split_key, used_ratio in zip(names, split_keys, ratio_values):
        if name not in root_dataset_cfg:
            raise ValueError(f"dataset.used_dataset includes '{name}' but dataset.{name} is not defined.")
        dataset_cfg = dict(root_dataset_cfg[name])
        dataset_cfg["used_ratio"] = used_ratio

        root_path = Path(dataset_cfg["root"])
        if not root_path.is_absolute():
            root_path = (PROJECT_ROOT / root_path).resolve()
        root = str(root_path)
        dataset = _build_single_dataset(name, dataset_cfg, root, split_key, img_size)
        datasets.append(_apply_used_ratio(dataset, dataset_cfg, name, split_key))

    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


def yolo_collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def create_dataloader(config, split="train"):
    _ = get_mode(config)
    dataset = build_dataset(config, split=split)
    dl_cfg = config["dataloader"]
    if isinstance(split, (list, tuple)):
        split_values = [str(v).strip().lower() for v in split if str(v).strip()]
        is_train_split = bool(split_values) and all(v == "train" for v in split_values)
    else:
        is_train_split = str(split).strip().lower() == "train"
    shuffle = dl_cfg["shuffle_train"] if is_train_split else dl_cfg["shuffle_eval"]
    return DataLoader(
        dataset,
        batch_size=dl_cfg["batch_size"],
        shuffle=shuffle,
        num_workers=dl_cfg["num_workers"],
        pin_memory=dl_cfg["pin_memory"],
        collate_fn=yolo_collate_fn,
    )
