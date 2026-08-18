from pathlib import Path

from methods.common.assets import AssetPreparationError
from tools.internal import prepare_pretrained


def test_prepare_pretrained_resolves_relative_output_and_offline_mode(
        tmp_path, monkeypatch, capsys):
    captured = {}

    def fake_prepare(destination, **kwargs):
        captured['destination'] = destination
        captured.update(kwargs)
        return Path(destination)

    monkeypatch.setattr(prepare_pretrained, 'PROJECT_ROOT', tmp_path)
    monkeypatch.setattr(prepare_pretrained, 'prepare_verified_asset', fake_prepare)

    result = prepare_pretrained.main([
        '--url', 'https://example.test/vgg16_caffe.pth',
        '--sha256', 'a' * 64,
        '--output', 'work_dirs/pretrained/vgg16_caffe.pth',
        '--offline',
    ])

    expected = tmp_path / 'work_dirs' / 'pretrained' / 'vgg16_caffe.pth'
    assert result == 0
    assert captured == {
        'destination': expected,
        'url': 'https://example.test/vgg16_caffe.pth',
        'expected_sha256': 'a' * 64,
        'allow_download': False,
    }
    assert capsys.readouterr().out.strip() == str(expected)


def test_prepare_pretrained_preserves_absolute_output(tmp_path, monkeypatch):
    captured = {}
    output = tmp_path / 'vgg16_caffe.pth'

    def fake_prepare(destination, **kwargs):
        captured['destination'] = destination
        return Path(destination)

    monkeypatch.setattr(prepare_pretrained, 'prepare_verified_asset', fake_prepare)

    result = prepare_pretrained.main([
        '--url', 'https://example.test/vgg16_caffe.pth',
        '--sha256', 'b' * 64,
        '--output', str(output),
    ])

    assert result == 0
    assert captured['destination'] == output


def test_prepare_pretrained_reports_asset_failure(monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise AssetPreparationError('checksum mismatch')

    monkeypatch.setattr(prepare_pretrained, 'prepare_verified_asset', fail)

    result = prepare_pretrained.main([
        '--url', 'https://example.test/vgg16_caffe.pth',
        '--sha256', 'c' * 64,
        '--output', 'work_dirs/pretrained/vgg16_caffe.pth',
    ])

    assert result == 1
    assert capsys.readouterr().err.strip() == 'error: checksum mismatch'
