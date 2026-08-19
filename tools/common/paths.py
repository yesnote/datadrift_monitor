'''Strict helpers for repository-relative configuration paths.'''

from pathlib import Path, PurePosixPath
from typing import Union


PathValue = Union[str, PurePosixPath]


def repository_root() -> Path:
    '''Return the ADAOD repository root without depending on the CWD.'''

    return Path(__file__).resolve().parents[2]


def repository_relative_path(value: PathValue) -> str:
    '''Validate and normalize a repository-relative POSIX path.'''

    raw_value = str(value)
    if not raw_value or chr(92) in raw_value:
        raise ValueError('repository paths must be non-empty POSIX paths')
    path = PurePosixPath(raw_value)
    if path.is_absolute() or path.parts[0].endswith(':'):
        raise ValueError('repository paths must be relative')
    if any(part in ('', '.', '..') for part in path.parts):
        raise ValueError('repository paths must not contain dot segments')
    return path.as_posix()

