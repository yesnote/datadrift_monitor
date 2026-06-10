import random
from contextlib import nullcontext
from functools import lru_cache

import numpy as np
import torch

from dataloaders.utils.data_utils import DATASET_CLASS_NAMES
from models.yolo.utils.general import coco80_to_coco91_class


@lru_cache(maxsize=1)
def coco91_to_80_pairs():
    return tuple((int(cat_id), int(i)) for i, cat_id in enumerate(coco80_to_coco91_class()))


_COCO91_TO_80_TENSOR_CACHE = {}


def coco91_to_80_lookup(device):
    device = torch.device(device)
    key = str(device)
    if key not in _COCO91_TO_80_TENSOR_CACHE:
        max_cat_id = max(cat_id for cat_id, _idx in coco91_to_80_pairs())
        lookup = torch.full((max_cat_id + 1,), -1, dtype=torch.long, device=device)
        for cat_id, idx in coco91_to_80_pairs():
            lookup[cat_id] = idx
        _COCO91_TO_80_TENSOR_CACHE[key] = lookup
    return _COCO91_TO_80_TENSOR_CACHE[key]


def map_coco91_to_80(labels, offset=0):
    lookup = coco91_to_80_lookup(labels.device)
    valid_id = (labels >= 0) & (labels < int(lookup.numel()))
    mapped = torch.full_like(labels, -1)
    if valid_id.any():
        mapped[valid_id] = lookup[labels[valid_id]]
    keep = mapped >= 0
    return mapped[keep] + int(offset), keep


def amp_enabled(config, device):
    return bool(config.get("training", {}).get("amp", True)) and torch.device(device).type == "cuda"


def autocast_context(device, enabled):
    if not enabled:
        return nullcontext()
    device_type = torch.device(device).type
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device_type, enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def make_grad_scaler(device, enabled):
    enabled = bool(enabled) and torch.device(device).type == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(torch.device(device).type, enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_matching_state_dict(model, state_dict):
    current = model.state_dict()
    filtered = {
        key: value
        for key, value in state_dict.items()
        if key in current and tuple(current[key].shape) == tuple(value.shape)
    }
    model.load_state_dict(filtered, strict=False)


def set_seed(seed):
    if seed is None:
        return
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def resolve_device(device_str):
    device_str = str(device_str or "cuda").strip().lower()
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    return torch.device(device_str)


def active_dataset_names(config):
    raw = config.get("dataset", {}).get("used_dataset", "")
    if isinstance(raw, str):
        return [raw.strip().lower()]
    if isinstance(raw, (list, tuple)):
        return [str(v).strip().lower() for v in raw if str(v).strip()]
    return []


def resolve_train_class_names(config):
    model_cfg = config.get("model", {})
    configured_names = model_cfg.get("class_names")
    if configured_names:
        if isinstance(configured_names, str):
            return [name.strip() for name in configured_names.split(",") if name.strip()]
        return [str(v) for v in configured_names]

    names = active_dataset_names(config)
    if names and all(name in DATASET_CLASS_NAMES for name in names):
        first = list(DATASET_CLASS_NAMES[names[0]])
        if all(list(DATASET_CLASS_NAMES[name]) == first for name in names):
            return first
        raise ValueError("Mixed datasets with different class spaces are not supported.")
    if any(name in DATASET_CLASS_NAMES for name in names):
        raise ValueError("Mixed custom datasets with non-matching class spaces are not supported.")
    return None


def trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]


def count_trainable_params(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def count_total_params(model):
    return int(sum(p.numel() for p in model.parameters()))


def training_options(config, device):
    training_cfg = config.get("training", {})
    return {
        "amp": bool(amp_enabled(config, device)),
        "freeze_feature_extractor": bool(training_cfg.get("freeze_feature_extractor", False)),
        "val_interval": max(1, int(training_cfg.get("val_interval", 1))),
        "save_last_interval": max(1, int(training_cfg.get("save_last_interval", 1))),
        "save_best": bool(training_cfg.get("save_best", True)),
        "save_optimizer": bool(training_cfg.get("save_optimizer", True)),
        "log_timing": bool(training_cfg.get("log_timing", False)),
        "grad_clip_norm": max(0.0, float(training_cfg.get("grad_clip_norm", 0.0))),
    }


def validate_training_config(config):
    training_cfg = config.get("training", {})
    epochs = int(training_cfg.get("epochs", 10))
    lr = float(training_cfg.get("lr", 1e-4))
    weight_decay = float(training_cfg.get("weight_decay", 1e-4))
    if epochs <= 0:
        raise ValueError("training.epochs must be >= 1.")
    if lr <= 0:
        raise ValueError("training.lr must be > 0.")
    if weight_decay < 0:
        raise ValueError("training.weight_decay must be >= 0.")
    return epochs, lr, weight_decay


def merge_epoch_timing(history_row, timing):
    if not timing:
        return history_row
    history_row.update({key: float(value) for key, value in timing.items()})
    return history_row
