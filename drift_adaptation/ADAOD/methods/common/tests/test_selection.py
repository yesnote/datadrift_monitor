import pytest

from methods.common.acquisition.selection import (
    AcquisitionScore,
    build_product_scores,
    select_top_k,
)
from methods.common.data.image_identity import SampleIdentity


def test_product_is_deterministic_and_empty_detection_is_zero() -> None:
    detected = AcquisitionScore(
        SampleIdentity('target', 'detected'),
        {'z': 2.0, 'a': 3.0},
        detection_count=1,
    )
    empty = AcquisitionScore(
        SampleIdentity('target', 'empty'),
        {'z': 100.0, 'a': 100.0},
        detection_count=0,
    )

    assert list(detected.components) == ['a', 'z']
    assert detected.final_score == 6.0
    assert empty.final_score == 0.0


def test_top_k_ties_are_broken_by_sample_id_then_namespace() -> None:
    scores = (
        AcquisitionScore(SampleIdentity('z-domain', 'same'), {'score': 1.0}, 1),
        AcquisitionScore(SampleIdentity('target', 'b'), {'score': 1.0}, 1),
        AcquisitionScore(SampleIdentity('target', 'a'), {'score': 1.0}, 1),
        AcquisitionScore(SampleIdentity('a-domain', 'same'), {'score': 1.0}, 1),
    )

    selected = select_top_k(scores, 4)

    assert [record.sample.qualified_id for record in selected] == [
        'target:a',
        'target:b',
        'a-domain:same',
        'z-domain:same',
    ]


def test_build_product_scores_requires_identical_sample_domains() -> None:
    first = SampleIdentity('target', 'a')
    second = SampleIdentity('target', 'b')
    scores = build_product_scores(
        {'fn': {second: 2.0, first: 1.0}},
        {first: 1, second: 0},
    )

    assert [score.sample for score in scores] == [first, second]
    assert [score.final_score for score in scores] == [1.0, 0.0]

    with pytest.raises(ValueError, match='same samples'):
        build_product_scores({'fn': {first: 1.0}}, {first: 1, second: 1})


def test_top_k_is_exact_and_rejects_duplicate_samples() -> None:
    sample = SampleIdentity('target', 'a')
    score = AcquisitionScore(sample, {'score': 1.0}, 1)

    assert select_top_k((score,), 0) == ()
    with pytest.raises(ValueError, match='more'):
        select_top_k((score,), 2)
    with pytest.raises(ValueError, match='duplicate'):
        select_top_k((score, score), 1)
