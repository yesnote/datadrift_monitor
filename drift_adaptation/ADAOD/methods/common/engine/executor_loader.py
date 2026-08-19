'''Executor-factory loading from an explicit method manifest module.'''

from importlib import import_module
from typing import Callable

from methods.common.contracts import MethodManifest

from .context import ExecutionContext
from .runner import StageExecutorRegistry


ExecutorRegistryFactory = Callable[[ExecutionContext], StageExecutorRegistry]


def load_executor_factory(manifest: MethodManifest) -> ExecutorRegistryFactory:
    '''Load the factory explicitly advertised by a method manifest.'''

    module_name = manifest.executor_module
    if not isinstance(module_name, str) or not module_name.strip():
        raise ValueError('method executor_module must be a non-empty string')
    module = import_module(module_name)
    factory = getattr(module, 'create_executor_registry', None)
    if factory is None:
        raise RuntimeError(
            'method {!r} executor module does not define '
            'create_executor_registry'.format(manifest.key)
        )
    if not callable(factory):
        raise TypeError(
            '{}.create_executor_registry must be callable'.format(module_name)
        )
    return factory
