import pytest
import torch
from torch import nn

from methods.ada_fnp.acquisition.mc_dropout import (
    collect_fixed_proposal_predictions, mc_dropout_enabled,
)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.dropout = nn.Dropout(0.5)

    def forward(self, value):
        return self.dropout(self.linear(value))


def test_mc_context_enables_only_dropout_and_restores_modes():
    model = ToyModel().train()
    with mc_dropout_enabled(model, [model.dropout]):
        assert not model.training
        assert not model.linear.training
        assert model.dropout.training
    assert model.training and model.linear.training and model.dropout.training


def test_fixed_proposal_callback_receives_same_tensor():
    proposals = torch.randn(3, 4)
    seen = []

    def predict(value):
        seen.append(value.data_ptr())
        return torch.softmax(torch.randn(3, 3), dim=1), value

    probabilities, boxes = collect_fixed_proposal_predictions(predict, proposals, 3)
    assert probabilities.shape == (3, 3, 3)
    assert boxes.shape == (3, 3, 4)
    assert seen == [proposals.data_ptr()] * 3


def test_mc_inference_requires_two_passes():
    with pytest.raises(ValueError):
        collect_fixed_proposal_predictions(lambda value: (value, value),
                                           torch.zeros(1, 4), 1)
