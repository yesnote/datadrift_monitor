'''Reusable layers for domain-adaptation models.'''

from .gradient_reversal import GradientReversal, gradient_reverse

__all__ = ['GradientReversal', 'gradient_reverse']
