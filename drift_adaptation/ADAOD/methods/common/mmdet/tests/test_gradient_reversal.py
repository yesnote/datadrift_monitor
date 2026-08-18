'''Tests for the gradient reversal primitive.'''

import pytest
import torch

from methods.common.mmdet.models.layers.gradient_reversal import (
    GradientReversal, gradient_reverse)


def test_gradient_reversal_is_identity_forward_and_scaled_negative_backward():
    inputs = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
    weights = torch.tensor([2.0, 3.0, -4.0])

    outputs = gradient_reverse(inputs, scale=0.25)
    (outputs * weights).sum().backward()

    torch.testing.assert_close(outputs, inputs.detach())
    torch.testing.assert_close(inputs.grad, -0.25 * weights)


def test_gradient_reversal_module_and_validation():
    layer = GradientReversal(scale=2.0)
    inputs = torch.ones(1, requires_grad=True)
    layer(inputs).sum().backward()

    torch.testing.assert_close(inputs.grad, torch.tensor([-2.0]))
    assert 'scale=2.0' in repr(layer)
    with pytest.raises(ValueError, match='non-negative'):
        GradientReversal(scale=-1.0)
