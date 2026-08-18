from collections import OrderedDict

import pytest

from methods.common.contracts import ExperimentPlan, MethodManifest
from tools.common.discovery import (
    discover_method_manifests,
    get_method_manifest,
)


def _manifest(key):
    return MethodManifest(
        key=key,
        api_version=1,
        description=key,
        config_factory=lambda: {'method': key},
        plan_factory=lambda config: ExperimentPlan(()),
    )


def test_discovery_bridge_sorts_manifest_keys():
    manifests = {'zeta': _manifest('zeta'), 'alpha': _manifest('alpha')}

    def discoverer():
        return manifests

    discovered = discover_method_manifests(discoverer=discoverer)
    assert isinstance(discovered, OrderedDict)
    assert tuple(discovered) == ('alpha', 'zeta')


def test_discovery_bridge_resolves_repository_manifest():
    manifest = get_method_manifest('ada-fnp')
    assert manifest.key == 'ada-fnp'
    assert manifest.api_version == 1


def test_discovery_bridge_reports_available_keys():
    def discoverer():
        return {'alpha': _manifest('alpha')}

    with pytest.raises(KeyError, match='available: alpha'):
        get_method_manifest('missing', discoverer=discoverer)

