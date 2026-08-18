from dataclasses import FrozenInstanceError
from typing import Dict, Mapping, Sequence

import pytest

from methods.common.data.annotations import reveal_acquired_annotations
from methods.common.data.image_identity import SampleIdentity
from methods.common.data.pool import PoolState, split_budget


def _samples(count: int = 12):
    return tuple(
        SampleIdentity('cityscapes', 'frame-{:03d}'.format(index))
        for index in range(count)
    )


class _Oracle:
    def __init__(self, annotations: Mapping[SampleIdentity, dict]) -> None:
        self.annotations = annotations
        self.calls = []

    def reveal(self, samples: Sequence[SampleIdentity]) -> Dict[SampleIdentity, dict]:
        self.calls.append(tuple(samples))
        return {sample: self.annotations[sample] for sample in samples}


def test_five_round_budget_is_exact_and_front_loads_remainder() -> None:
    assert split_budget(12) == (3, 3, 2, 2, 2)
    assert split_budget(2) == (1, 1, 0, 0, 0)
    assert sum(split_budget(101)) == 101


def test_pool_transitions_exactly_five_rounds_without_reselection() -> None:
    samples = _samples()
    state = PoolState.initialize(samples, total_budget=12)
    cursor = 0

    for expected_budget in (3, 3, 2, 2, 2):
        assert state.next_budget == expected_budget
        state = state.acquire(samples[cursor : cursor + expected_budget])
        cursor += expected_budget

    assert state.is_complete
    assert state.labeled == samples
    assert state.unlabeled == ()
    assert PoolState.from_dict(state.to_dict()) == state
    with pytest.raises(RuntimeError, match='five'):
        state.acquire(())


def test_pool_rejects_wrong_budget_duplicates_and_reselection() -> None:
    samples = _samples()
    state = PoolState.initialize(samples, total_budget=10)

    with pytest.raises(ValueError, match='exactly 2'):
        state.acquire(samples[:1])
    with pytest.raises(ValueError, match='duplicate'):
        state.acquire((samples[0], samples[0]))

    state = state.acquire(samples[:2])
    with pytest.raises(ValueError, match='previously selected'):
        state.acquire((samples[0], samples[2]))


def test_pool_is_frozen_and_validates_total_budget() -> None:
    samples = _samples(3)
    state = PoolState.initialize(samples, total_budget=3)

    with pytest.raises(FrozenInstanceError):
        state.labeled = samples  # type: ignore[misc]
    with pytest.raises(ValueError, match='exceeds'):
        PoolState.initialize(samples, total_budget=4)
    with pytest.raises(TypeError, match='SampleIdentity'):
        PoolState.initialize(('not-an-identity',), total_budget=0)


def test_oracle_reveals_only_the_newly_acquired_samples() -> None:
    samples = _samples(5)
    previous = PoolState.initialize(samples, total_budget=5)
    current = previous.acquire((samples[3],))
    oracle = _Oracle({sample: {'id': sample.sample_id} for sample in samples})

    revealed = reveal_acquired_annotations(oracle, previous, current)

    assert tuple(revealed) == (samples[3],)
    assert oracle.calls == [(samples[3],)]
    with pytest.raises(TypeError):
        revealed[samples[0]] = {}  # type: ignore[index]


def test_oracle_result_must_match_the_acquisition_exactly() -> None:
    samples = _samples(5)
    previous = PoolState.initialize(samples, total_budget=5)
    current = previous.acquire((samples[0],))

    class WrongOracle:
        def reveal(self, requested):
            del requested
            return {samples[1]: {'id': samples[1].sample_id}}

    with pytest.raises(ValueError, match='wrong sample set'):
        reveal_acquired_annotations(WrongOracle(), previous, current)


def test_zero_budget_round_does_not_touch_the_oracle() -> None:
    samples = _samples(2)
    state = PoolState.initialize(samples, total_budget=2)
    state = state.acquire((samples[0],))
    state = state.acquire((samples[1],))
    previous = state
    current = previous.acquire(())
    oracle = _Oracle({})

    assert dict(reveal_acquired_annotations(oracle, previous, current)) == {}
    assert oracle.calls == []
