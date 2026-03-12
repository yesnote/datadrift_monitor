import torch


def to_device(batch_targets, device):
    result = []
    for target in batch_targets:
        item = {}
        for key, value in target.items():
            item[key] = value.to(device) if isinstance(value, torch.Tensor) else value
        result.append(item)
    return result
