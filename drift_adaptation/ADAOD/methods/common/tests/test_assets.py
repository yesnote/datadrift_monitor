import hashlib

import pytest

from methods.common.assets import (
    AssetPreparationError,
    AssetVerificationError,
    prepare_verified_asset,
    sha256_file,
)


ASSET_URL = 'https://example.test/vgg16_caffe.pth'


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_sha256_file_streams_and_validates_chunk_size(tmp_path):
    path = tmp_path / 'asset.bin'
    payload = b'0123456789' * 100
    path.write_bytes(payload)

    assert sha256_file(path, chunk_size=7) == _digest(payload)
    with pytest.raises(ValueError, match='chunk_size must be positive'):
        sha256_file(path, chunk_size=0)


def test_existing_verified_asset_is_reused_without_download(tmp_path):
    path = tmp_path / 'vgg16_caffe.pth'
    payload = b'verified'
    path.write_bytes(payload)

    prepared = prepare_verified_asset(
        path,
        url=ASSET_URL,
        expected_sha256=_digest(payload),
        downloader=lambda *args: pytest.fail('verified asset was downloaded'),
    )

    assert prepared == path.resolve()
    assert path.read_bytes() == payload


def test_invalid_existing_asset_is_not_overwritten(tmp_path):
    path = tmp_path / 'vgg16_caffe.pth'
    path.write_bytes(b'corrupt')

    with pytest.raises(AssetVerificationError, match='cached asset SHA-256'):
        prepare_verified_asset(
            path,
            url=ASSET_URL,
            expected_sha256=_digest(b'expected'),
            downloader=lambda *args: pytest.fail('invalid cache was overwritten'),
        )

    assert path.read_bytes() == b'corrupt'


def test_download_is_verified_and_atomically_installed(tmp_path):
    path = tmp_path / 'nested' / 'vgg16_caffe.pth'
    payload = b'downloaded checkpoint'
    calls = []

    def downloader(url, destination):
        calls.append((url, destination))
        destination.write_bytes(payload)

    prepared = prepare_verified_asset(
        path,
        url=ASSET_URL,
        expected_sha256=_digest(payload),
        downloader=downloader,
    )

    assert prepared == path.resolve()
    assert path.read_bytes() == payload
    assert calls[0][0] == ASSET_URL
    assert calls[0][1].parent == path.parent
    assert not tuple(path.parent.glob('*.part'))


def test_failed_download_leaves_no_destination_or_partial_file(tmp_path):
    path = tmp_path / 'vgg16_caffe.pth'

    def downloader(url, destination):
        destination.write_bytes(b'wrong payload')

    with pytest.raises(AssetVerificationError, match='downloaded asset SHA-256'):
        prepare_verified_asset(
            path,
            url=ASSET_URL,
            expected_sha256=_digest(b'expected payload'),
            downloader=downloader,
        )

    assert not path.exists()
    assert not tuple(tmp_path.glob('*.part'))


def test_missing_offline_asset_and_invalid_sources_fail_explicitly(tmp_path):
    path = tmp_path / 'vgg16_caffe.pth'
    with pytest.raises(AssetPreparationError, match='downloads are disabled'):
        prepare_verified_asset(
            path,
            url=ASSET_URL,
            expected_sha256='0' * 64,
            allow_download=False,
        )
    with pytest.raises(ValueError, match='absolute HTTPS'):
        prepare_verified_asset(
            path,
            url='http://example.test/checkpoint.pth',
            expected_sha256='0' * 64,
        )
    with pytest.raises(ValueError, match='64 hexadecimal'):
        prepare_verified_asset(
            path,
            url=ASSET_URL,
            expected_sha256='not-a-digest',
        )
