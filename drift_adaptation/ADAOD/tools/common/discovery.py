'''CLI-side bridge to deterministic method-manifest discovery.'''

from collections import OrderedDict
from pathlib import Path
from typing import Callable, Mapping, Optional

from methods.common.contracts import MethodManifest
from methods.common.registry import discover_methods


ManifestDiscoverer = Callable[..., Mapping[str, MethodManifest]]


def discover_method_manifests(
    methods_root: Optional[Path] = None,
    discoverer: ManifestDiscoverer = discover_methods,
) -> Mapping[str, MethodManifest]:
    '''Return discovered manifests ordered by their public method key.'''

    if methods_root is None:
        manifests = discoverer()
    else:
        manifests = discoverer(methods_root=methods_root)
    return OrderedDict((key, manifests[key]) for key in sorted(manifests))


def get_method_manifest(
    key: str,
    methods_root: Optional[Path] = None,
    discoverer: ManifestDiscoverer = discover_methods,
) -> MethodManifest:
    '''Resolve one manifest without maintaining a second method list.'''

    manifests = discover_method_manifests(methods_root, discoverer)
    try:
        return manifests[key]
    except KeyError as error:
        choices = ', '.join(manifests)
        raise KeyError(
            f'unknown method {key!r}; available: {choices}'
        ) from error

