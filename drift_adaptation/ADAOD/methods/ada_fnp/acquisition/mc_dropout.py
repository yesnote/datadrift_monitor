'''Fixed-proposal Monte Carlo Dropout utilities.'''

from contextlib import contextmanager
from typing import Iterator, Sequence

from torch import nn


@contextmanager
def monte_carlo_dropout_enabled(
    model: nn.Module, dropout_modules: Sequence[nn.Module]
) -> Iterator[None]:
    '''Keep the model in eval mode while enabling only selected dropout.'''

    states = [(module, module.training) for module in model.modules()]
    selected = {id(module) for module in dropout_modules}
    if not selected:
        raise ValueError('at least one dropout module is required')
    if not selected.issubset({id(module) for module, _ in states}):
        raise ValueError('dropout modules must belong to model')
    try:
        model.eval()
        for module in dropout_modules:
            if not isinstance(module, nn.modules.dropout._DropoutNd):
                raise TypeError('selected module is not a dropout module')
            module.train(True)
        yield
    finally:
        for module, training in states:
            module.training = training
