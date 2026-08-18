"""Utilities for preparing immutable, checksum-verified remote assets."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
import urllib.request
from pathlib import Path
from typing import Callable, Optional, Union
from urllib.parse import urlparse


PathLike = Union[str, os.PathLike]
Downloader = Callable[[str, Path], None]
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


class AssetPreparationError(RuntimeError):
    """Raised when a required asset cannot be prepared safely."""


class AssetVerificationError(AssetPreparationError):
    """Raised when an asset does not match its pinned digest."""


def sha256_file(path: PathLike, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_source(url: str, expected_sha256: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme.lower() != "https" or not parsed.netloc:
        raise ValueError("asset URL must be an absolute HTTPS URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("asset URL must not contain credentials")
    if not _SHA256_PATTERN.fullmatch(expected_sha256):
        raise ValueError("expected_sha256 must contain exactly 64 hexadecimal characters")
    return expected_sha256.lower()


def _download_https(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "ADAOD-asset-preparer/1"})
    try:
        with urllib.request.urlopen(request) as response, destination.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
    except Exception as exc:
        raise AssetPreparationError("failed to download asset from {!r}".format(url)) from exc


def prepare_verified_asset(
    destination: PathLike,
    *,
    url: str,
    expected_sha256: str,
    allow_download: bool = True,
    downloader: Optional[Downloader] = None,
) -> Path:
    """Ensure that ``destination`` contains the pinned asset.

    Existing files are verified before reuse. New downloads are written beside
    the destination, verified, and atomically installed. An invalid existing
    file is never silently overwritten.
    """

    expected_digest = _validate_source(url, expected_sha256)
    destination_path = Path(destination).expanduser().resolve()

    if destination_path.exists():
        if not destination_path.is_file():
            raise AssetPreparationError(
                "asset destination is not a file: {!s}".format(destination_path)
            )
        actual_digest = sha256_file(destination_path)
        if actual_digest != expected_digest:
            raise AssetVerificationError(
                "cached asset SHA-256 mismatch for {!s}: expected {}, got {}".format(
                    destination_path, expected_digest, actual_digest
                )
            )
        return destination_path

    if not allow_download:
        raise AssetPreparationError(
            "asset is missing and downloads are disabled: {!s}".format(destination_path)
        )

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_fd, temporary_name = tempfile.mkstemp(
        dir=str(destination_path.parent),
        prefix=".{}-".format(destination_path.name),
        suffix=".part",
    )
    os.close(temporary_fd)
    temporary_path = Path(temporary_name)
    download = downloader or _download_https

    try:
        download(url, temporary_path)
        if not temporary_path.is_file():
            raise AssetPreparationError("asset downloader did not create a file")
        actual_digest = sha256_file(temporary_path)
        if actual_digest != expected_digest:
            raise AssetVerificationError(
                "downloaded asset SHA-256 mismatch: expected {}, got {}".format(
                    expected_digest, actual_digest
                )
            )
        os.replace(temporary_path, destination_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return destination_path
