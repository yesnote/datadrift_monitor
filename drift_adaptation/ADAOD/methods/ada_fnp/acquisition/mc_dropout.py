'''Fixed-proposal Monte Carlo Dropout utilities.'''

from contextlib import contextmanager
from typing import Callable, Iterator, Sequence, Tuple

import torch
from torch import nn


@contextmanager
def mc_dropout_enabled(
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


def collect_fixed_proposal_predictions(
    predict_roi: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    proposals: torch.Tensor,
    passes: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    '''Run a stochastic RoI callback repeatedly on the exact same proposals.'''

    if passes < 2:
        raise ValueError('Monte Carlo inference requires at least two passes')
    probabilities = []
    boxes = []
    for _ in range(passes):
        pass_probabilities, pass_boxes = predict_roi(proposals)
        probabilities.append(pass_probabilities)
        boxes.append(pass_boxes)
    probability_samples = torch.stack(probabilities)
    box_samples = torch.stack(boxes)
    if probability_samples.shape[1] != len(proposals):
        raise ValueError('RoI probability count does not match fixed proposals')
    if box_samples.shape[1] != len(proposals):
        raise ValueError('RoI box count does not match fixed proposals')
    return probability_samples, box_samples
