'''Checkpoint helpers that preserve stochastic training state.'''

import random
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import torch


def capture_rng_state() -> Dict[str, Any]:
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if 'cuda' in state:
        if not torch.cuda.is_available():
            raise RuntimeError('checkpoint contains CUDA RNG state but CUDA is unavailable')
        torch.cuda.set_rng_state_all(state['cuda'])


def save_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def load_checkpoint(path: Path) -> Dict[str, Any]:
    return torch.load(path, map_location='cpu', weights_only=False)
