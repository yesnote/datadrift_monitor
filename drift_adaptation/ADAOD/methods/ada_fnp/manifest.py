'''Manifest discovered by the project method registry.'''

from methods.common.contracts import MethodManifest

from .configs.default import get_config
from .plan import build_plan


MANIFEST = MethodManifest(
    key='ada-fnp',
    api_version=1,
    description='Active Domain Adaptation with False Negative Prediction',
    config_factory=get_config,
    plan_factory=build_plan,
    custom_imports=(
        'methods.common.mmdet.registration',
        'methods.ada_fnp.registration',
    ),
)
