'''Trusted PyTorch checkpoint file I/O.'''

from pathlib import Path
from typing import Any, Dict, Mapping

import torch


def save_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def load_checkpoint(path: Path) -> Dict[str, Any]:
    return torch.load(path, map_location='cpu', weights_only=False)
