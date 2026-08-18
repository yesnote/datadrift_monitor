from copy import deepcopy

import pytest

from tools.common.config import compose_config, config_fingerprint
from tools.common.discovery import get_method_manifest


def test_composition_uses_catalogs_and_preserves_method_defaults():
    manifest = get_method_manifest('ada-fnp')
    method_defaults = manifest.config_factory()
    config = compose_config(manifest)
    assert config['scenario'] == 'cityscapes-to-foggy'
    assert config['dataset']['source']['image_root'] == 'data/leftImg8bit'
    assert config['dataset']['target']['beta'] == 0.02
    assert config['dataset']['target']['train_annotation_access'] == 'oracle_only'
    assert config['detector']['name'] == 'faster-rcnn-vgg16'
    assert config['detector']['batch_normalization'] is False
    assert config['detector']['dropout_probability'] == (
        method_defaults['detector']['dropout_probability']
    )
    assert config['training'] == method_defaults['training']
    assert config['fnpm'] == method_defaults['fnpm']
    assert config['runtime']['work_root'] == 'work_dirs'


def test_composition_does_not_mutate_method_defaults_or_overrides():
    manifest = get_method_manifest('ada-fnp')
    before = manifest.config_factory()
    overrides = {'seed': 7, 'acquisition': {'budget_percent': 5.0}}
    original_overrides = deepcopy(overrides)
    config = compose_config(manifest, overrides=overrides)
    assert config['seed'] == 7
    assert config['acquisition']['budget_percent'] == 5.0
    assert manifest.config_factory() == before
    assert overrides == original_overrides


def test_config_fingerprint_is_stable_and_sensitive_to_changes():
    manifest = get_method_manifest('ada-fnp')
    first = compose_config(manifest)
    second = compose_config(manifest)
    assert config_fingerprint(first) == config_fingerprint(second)
    second['seed'] += 1
    assert config_fingerprint(first) != config_fingerprint(second)


def test_composition_rejects_unknown_catalog_keys():
    manifest = get_method_manifest('ada-fnp')
    with pytest.raises(KeyError, match='unknown dataset scenario'):
        compose_config(manifest, dataset_key='missing')
    with pytest.raises(KeyError, match='unknown detector'):
        compose_config(manifest, detector_key='missing')
    with pytest.raises(KeyError, match='unknown runtime'):
        compose_config(manifest, runtime_key='missing')
