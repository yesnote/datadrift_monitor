from pathlib import Path

from torch.utils.data import ConcatDataset, DataLoader
import yaml

from dataloaders.datasets.coco import COCODataset
from dataloaders.datasets.null_image import NullImageDataset
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

    if name == "null_image":
        if "num_samples" not in dataset_cfg:
            raise ValueError("dataset.null_image.num_samples is required.")
        return NullImageDataset(
            num_samples=dataset_cfg["num_samples"],
            img_size=img_size,
            seed=dataset_cfg.get("seed"),
        )

    raise ValueError(f"Unsupported dataset name: {name}")


def build_dataset(config, split="train"):
    mode = get_mode(config)
    root_dataset_cfg = config["dataset"]
    names = _normalize_dataset_names(root_dataset_cfg)

    if isinstance(split, (list, tuple)):
        split_keys = [str(v).strip() for v in split if str(v).strip()]
    else:
        split_keys = [str(split).strip()] * len(names)

    if len(split_keys) != len(names):
        raise ValueError(
            f"Length mismatch: used_dataset has {len(names)} entries but split has {len(split_keys)} entries."
        )

    datasets = []
    img_size = config["model"]["img_size"]
    for name, split_key in zip(names, split_keys):
        if name not in root_dataset_cfg:
            raise ValueError(f"dataset.used_dataset includes '{name}' but dataset.{name} is not defined.")
        dataset_cfg = root_dataset_cfg[name]

        if name == "null_image" and mode != "predict":
            raise ValueError("dataset.null_image is supported only when mode='predict'.")

        root = None
        if name != "null_image":
            root_path = Path(dataset_cfg["root"])
            if not root_path.is_absolute():
                root_path = (PROJECT_ROOT / root_path).resolve()
            root = str(root_path)
        datasets.append(_build_single_dataset(name, dataset_cfg, root, split_key, img_size))

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
