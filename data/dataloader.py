from torch.utils.data import DataLoader
from data.dataset import LPDataset

def build_dataloader(cfg):
    dataset = LPDataset(cfg, split="val")
    return DataLoader(
        dataset,
        batch_size=cfg['dataloader']['batch_size'],
        shuffle=False,
        num_workers=cfg["dataloader"]["num_workers"]
    )
