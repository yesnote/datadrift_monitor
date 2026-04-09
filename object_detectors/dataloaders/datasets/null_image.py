import torch
from torch.utils.data import Dataset


class NullImageDataset(Dataset):
    def __init__(self, num_samples, img_size=640, seed=None):
        self.num_samples = int(num_samples)
        if self.num_samples <= 0:
            raise ValueError("dataset.null_image.num_samples must be >= 1.")

        if isinstance(img_size, int):
            h = w = int(img_size)
        elif isinstance(img_size, (list, tuple)) and len(img_size) == 2:
            h, w = int(img_size[0]), int(img_size[1])
        else:
            raise ValueError("model.img_size must be an int or length-2 list/tuple.")
        if h <= 0 or w <= 0:
            raise ValueError("model.img_size must be positive.")
        self.height = h
        self.width = w

        self.seed = None if seed is None else int(seed)

    def __len__(self):
        return self.num_samples

    def _make_noise_image(self, index):
        if self.seed is None:
            return torch.rand((3, self.height, self.width), dtype=torch.float32)
        generator = torch.Generator()
        generator.manual_seed(self.seed + int(index))
        return torch.rand((3, self.height, self.width), generator=generator, dtype=torch.float32)

    def __getitem__(self, index):
        image = self._make_noise_image(index)
        target = {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([int(index)], dtype=torch.int64),
            "path": f"null_image://{int(index)}",
            "dataset_name": "null_image",
            "gt_class_names": [],
        }
        return image, target
