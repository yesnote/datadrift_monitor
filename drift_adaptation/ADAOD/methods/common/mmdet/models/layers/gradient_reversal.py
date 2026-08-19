'''Gradient reversal with an identity forward pass.'''

from torch import Tensor
from torch.autograd import Function


class _GradientReverseFunction(Function):

    @staticmethod
    def forward(ctx, inputs: Tensor, scale: float) -> Tensor:
        ctx.scale = float(scale)
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return grad_output.neg().mul(ctx.scale), None


def gradient_reverse(inputs: Tensor, scale: float = 1.0) -> Tensor:
    '''Return ``inputs`` while multiplying its backward gradient by ``-scale``.'''
    if scale < 0:
        raise ValueError('gradient reversal scale must be non-negative')
    return _GradientReverseFunction.apply(inputs, float(scale))
