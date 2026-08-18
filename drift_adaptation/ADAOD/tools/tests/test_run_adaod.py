import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from methods.ada_fnp.manifest import MANIFEST
from tools.common.config import compose_config
from tools.run_adaod import _prepare_run


def test_cli_sets_deterministic_cublas_workspace_before_execution():
    environment = os.environ.copy()
    environment.pop('CUBLAS_WORKSPACE_CONFIG', None)
    completed = subprocess.run(
        [
            sys.executable,
            '-c',
            'import os; import tools.run_adaod; '
            'print(os.environ[\'CUBLAS_WORKSPACE_CONFIG\'])',
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == ':4096:8'


def test_run_adaod_supports_direct_script_dry_run():
    repository_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            str(repository_root / 'tools' / 'run_adaod.py'),
            '--method',
            'ada-fnp',
            '--dataset',
            'cityscapes-to-foggy',
            '--detector',
            'faster-rcnn-vgg16',
            '--dry-run',
        ],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload['config']['method'] == 'ada-fnp'
    assert len(payload['plan']['stages']) == 29


def test_prepare_run_requires_explicit_resume_and_matching_config(tmp_path):
    config = compose_config(MANIFEST)
    run_directory = tmp_path / 'run'

    state_store = _prepare_run(config, run_directory, resume=False)
    assert state_store.path.is_file()

    with pytest.raises(SystemExit, match='--resume'):
        _prepare_run(config, run_directory, resume=False)
    state_store.path.unlink()
    resumed = _prepare_run(config, run_directory, resume=True)
    assert resumed.path == state_store.path
    assert resumed.path.is_file()

    changed = dict(config)
    changed['seed'] = 99
    with pytest.raises(SystemExit, match='differs'):
        _prepare_run(changed, run_directory, resume=True)


def test_prepare_run_persists_resolved_config_before_execution(tmp_path):
    config = compose_config(MANIFEST)
    run_directory = tmp_path / 'run'

    _prepare_run(config, run_directory, resume=False)

    stored = json.loads(
        (run_directory / 'resolved_config.json').read_text(encoding='utf-8')
    )
    assert stored['config']['method'] == 'ada-fnp'
    assert len(stored['config_fingerprint']) == 64
