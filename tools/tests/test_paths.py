import pytest

from tools.common.paths import (
    repository_relative_path,
    repository_root,
    resolve_repository_path,
)


def test_repository_root_is_independent_of_current_working_directory():
    assert repository_root().name == 'ADAOD'
    assert (repository_root() / 'methods').is_dir()


def test_repository_relative_paths_resolve_beneath_requested_root(tmp_path):
    assert repository_relative_path('data/leftImg8bit') == 'data/leftImg8bit'
    assert resolve_repository_path('work_dirs/run', tmp_path) == (
        tmp_path / 'work_dirs' / 'run'
    ).resolve()


@pytest.mark.parametrize(
    'value',
    ('', '/absolute/path', '../outside', 'data/../outside', r'data\images', 'C:/data'),
)
def test_repository_relative_paths_reject_unsafe_values(value):
    with pytest.raises(ValueError):
        repository_relative_path(value)
