from dataloaders.common import create_detection_dataloader
from dataloaders.factory import build_dataset, detection_collate_fn


def create_dataloader(config, split="train"):
    return create_detection_dataloader(config, split=split, default_aspect_ratio_grouping=False)


__all__ = ["build_dataset", "create_dataloader", "detection_collate_fn"]
