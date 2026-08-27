'''Manifest discovered by the project method registry.'''

from methods.common.contracts import MethodManifest

from .configs.default import get_config
from .plan import build_plan


MANIFEST = MethodManifest(
    key='aada',
    api_version=2,
    description='Active Adversarial Domain Adaptation under ADA-FNP protocol',
    config_factory=get_config,
    plan_factory=build_plan,
    executor_module='methods.aada.execution.stages',
)
