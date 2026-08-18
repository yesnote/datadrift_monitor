from pathlib import Path

import pytest

from tools.internal import prepare_cityscapes


def test_cli_resolves_only_repository_relative_dataset_paths(
    tmp_path, monkeypatch, capsys
) -> None:
    captured = {}

    def fake_prepare(clear, foggy, polygons, cache, root, **kwargs):
        captured.update(
            clear=clear,
            foggy=foggy,
            polygons=polygons,
            cache=cache,
            root=root,
            kwargs=kwargs,
        )
        return {'fingerprint': 'a' * 64}

    monkeypatch.setattr(prepare_cityscapes, 'prepare_cityscapes_to_foggy', fake_prepare)

    assert prepare_cityscapes.main([], repository_root_path=tmp_path) == 0
    assert captured['clear'] == (
        tmp_path / 'data' / 'Cityscapes' / 'leftImg8bit'
    )
    assert captured['foggy'] == (
        tmp_path / 'data' / 'Cityscapes' / 'leftImg8bit_foggy'
    )
    assert captured['polygons'] == tmp_path / 'data' / 'Cityscapes' / 'gtFine'
    assert captured['cache'] == (
        tmp_path / 'work_dirs' / '.dataset_cache' / 'cityscapes-to-foggy'
    )
    assert 'fingerprint' in capsys.readouterr().out


@pytest.mark.parametrize(
    ('argument', 'value'),
    [
        ('--clear-images', 'C:/datasets/Cityscapes/leftImg8bit'),
        ('--foggy-images', '../leftImg8bit_foggy'),
        ('--polygons', 'external/gtFine'),
        ('--clear-images', 'data/leftImg8bit'),
        ('--cache-directory', '../cache'),
    ],
)
def test_cli_rejects_non_repository_or_non_data_paths(
    tmp_path: Path, argument: str, value: str
) -> None:
    with pytest.raises(SystemExit):
        prepare_cityscapes.main(
            [argument, value],
            repository_root_path=tmp_path,
            expected_train_images=0,
            expected_val_images=0,
        )
