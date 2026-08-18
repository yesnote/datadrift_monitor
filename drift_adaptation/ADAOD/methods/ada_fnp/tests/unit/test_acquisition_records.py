import pytest

from methods.ada_fnp.acquisition.records import RawAdaFnpScore, normalize_scores
from methods.common.data.image_identity import SampleIdentity


def _record(index, detection_count=1):
    return RawAdaFnpScore(
        SampleIdentity('foggy-cityscapes.beta-0.02.train', f'scene_{index}'),
        false_negative=float(index), localization=float(index),
        entropy=float(index), diversity=float(index),
        source_domain_probability=0.5, detection_count=detection_count,
    )


def test_constant_components_are_neutral():
    records = [_record(1), _record(1)]
    records[1] = RawAdaFnpScore(
        SampleIdentity('foggy-cityscapes.beta-0.02.train', 'different'),
        1., 1., 1., 1., .5, 1,
    )
    scores = normalize_scores(records)
    assert all(score.final_score == pytest.approx(.5 ** 4) for score in scores)


def test_empty_detection_forces_final_zero_after_normalization():
    records = [_record(0, detection_count=0), _record(2)]
    scores = {score.sample.sample_id: score for score in normalize_scores(records)}
    assert scores['scene_0'].final_score == 0


def test_empty_detection_rejects_nonzero_box_scores():
    with pytest.raises(ValueError):
        RawAdaFnpScore(SampleIdentity('target', 'sample'), 1., 1., 0., 1., .5, 0)
