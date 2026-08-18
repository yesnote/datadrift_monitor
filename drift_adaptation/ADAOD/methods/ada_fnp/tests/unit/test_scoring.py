import math

import pytest
import torch

from methods.ada_fnp.acquisition.scoring import (
    domain_diversity, foreground_entropy, localization_score,
)


def test_entropy_excludes_background_without_renormalizing():
    probabilities = torch.tensor([[0.25, 0.75, 0.0]])
    expected = -0.25 * math.log(0.25) - 0.75 * math.log(0.75)
    assert foreground_entropy(probabilities).item() == pytest.approx(expected)


def test_localization_score_is_scale_invariant():
    boxes = torch.tensor([
        [[0, 0, 10, 10]],
        [[2, 0, 12, 10]],
    ], dtype=torch.float32)
    small = localization_score(boxes, 20, 20)
    large = localization_score(boxes * 2, 40, 40)
    assert small.item() == pytest.approx(large.item())


def test_empty_boxes_have_zero_location_score():
    boxes = torch.empty((10, 0, 4))
    assert localization_score(boxes, 20, 20).item() == 0


def test_domain_diversity_clamps_zero_probability():
    result = domain_diversity(torch.tensor([0.0]))
    assert torch.isfinite(result).all()
    assert result.item() > 0


def test_domain_diversity_rejects_non_probability():
    with pytest.raises(ValueError):
        domain_diversity(torch.tensor([1.1]))
