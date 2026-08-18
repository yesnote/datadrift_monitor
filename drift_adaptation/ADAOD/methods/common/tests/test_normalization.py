import math

import pytest

from methods.common.acquisition.normalization import (
    lower_clamped_standardize,
    standardize_components,
)
from methods.common.data.image_identity import SampleIdentity


def test_eq10_style_population_standardization_and_lower_clamp() -> None:
    standardized = lower_clamped_standardize((1.0, 2.0, 3.0))

    assert standardized[0] == pytest.approx(0.5 - math.sqrt(1.5) / 6.0)
    assert standardized[1] == pytest.approx(0.5)
    assert standardized[2] == pytest.approx(0.5 + math.sqrt(1.5) / 6.0)

    outlier = lower_clamped_standardize((0.0,) + (10.0,) * 16)
    assert outlier[0] == 0.0


def test_constant_component_maps_to_neutral_half() -> None:
    assert lower_clamped_standardize((7.0, 7.0, 7.0)) == (0.5, 0.5, 0.5)


def test_empty_and_non_finite_inputs_are_explicit() -> None:
    assert lower_clamped_standardize(()) == ()
    with pytest.raises(ValueError, match='finite'):
        lower_clamped_standardize((1.0, float('inf')))


def test_components_require_the_same_sample_domain() -> None:
    first = SampleIdentity('target', 'a')
    second = SampleIdentity('target', 'b')

    result = standardize_components(
        {
            'fn': {first: 2.0, second: 2.0},
            'localization': {first: 1.0, second: 3.0},
        }
    )
    assert result['fn'] == {first: 0.5, second: 0.5}
    assert result['localization'][first] == pytest.approx(1.0 / 3.0)

    with pytest.raises(ValueError, match='same sample set'):
        standardize_components({'a': {first: 1.0}, 'b': {second: 1.0}})
