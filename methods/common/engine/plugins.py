'''Executor-factory discovery through method manifest imports.'''

from importlib import import_module
from typing import Callable

from methods.common.contracts import MethodManifest

from .context import ExecutionContext
from .runner import StageExecutorRegistry


ExecutorRegistryFactory = Callable[[ExecutionContext], StageExecutorRegistry]


def load_executor_factory(manifest: MethodManifest) -> ExecutorRegistryFactory:
    '''Load the one factory explicitly advertised by a method manifest.'''

    for module_name in manifest.custom_imports:
        module = import_module(module_name)
        factory = getattr(module, 'create_executor_registry', None)
        if factory is not None:
            if not callable(factory):
                raise TypeError(
                    '{}.create_executor_registry must be callable'.format(module_name)
                )
            return factory
    raise RuntimeError(
        'method {!r} does not advertise an executor factory'.format(manifest.key)
    )
