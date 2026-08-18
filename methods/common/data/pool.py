'''Immutable active-target pool state and exact round-budget transitions.'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence, Tuple

from .image_identity import SampleIdentity


ACTIVE_ROUNDS = 5


def _require_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError('{} must be an integer'.format(name))


def _require_unique(name: str, samples: Sequence[SampleIdentity]) -> None:
    if len(samples) != len(set(samples)):
        raise ValueError('{} contains duplicate sample identities'.format(name))


def _require_identities(name: str, samples: Sequence[SampleIdentity]) -> None:
    if any(not isinstance(sample, SampleIdentity) for sample in samples):
        raise TypeError('{} must contain only SampleIdentity values'.format(name))


def split_budget(total_budget: int, rounds: int = ACTIVE_ROUNDS) -> Tuple[int, ...]:
    '''Split an integer budget exactly, assigning the remainder to early rounds.'''

    _require_integer('total_budget', total_budget)
    _require_integer('rounds', rounds)
    if total_budget < 0:
        raise ValueError('total_budget must not be negative')
    if rounds <= 0:
        raise ValueError('rounds must be positive')
    quotient, remainder = divmod(total_budget, rounds)
    return tuple(
        quotient + (1 if round_offset < remainder else 0)
        for round_offset in range(rounds)
    )


@dataclass(frozen=True)
class AcquisitionRound:
    '''One exact-budget transition from target-unlabeled to target-labeled.'''

    round_index: int
    budget: int
    selected: Tuple[SampleIdentity, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, 'selected', tuple(self.selected))
        _require_integer('round_index', self.round_index)
        _require_integer('budget', self.budget)
        if self.round_index < 1 or self.round_index > ACTIVE_ROUNDS:
            raise ValueError('round_index must be between 1 and 5')
        if self.budget < 0:
            raise ValueError('round budget must not be negative')
        if len(self.selected) != self.budget:
            raise ValueError('selected sample count must equal the round budget')
        _require_identities('round selection', self.selected)
        _require_unique('round selection', self.selected)

    def to_dict(self) -> dict:
        return {
            'round_index': self.round_index,
            'budget': self.budget,
            'selected': [sample.to_dict() for sample in self.selected],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> 'AcquisitionRound':
        return cls(
            round_index=value['round_index'],
            budget=value['budget'],
            selected=tuple(
                SampleIdentity.from_dict(sample) for sample in value['selected']
            ),
        )


@dataclass(frozen=True)
class PoolState:
    '''Validated target pool state for a fixed five-round acquisition plan.'''

    universe: Tuple[SampleIdentity, ...]
    labeled: Tuple[SampleIdentity, ...]
    unlabeled: Tuple[SampleIdentity, ...]
    round_budgets: Tuple[int, ...]
    history: Tuple[AcquisitionRound, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, 'universe', tuple(self.universe))
        object.__setattr__(self, 'labeled', tuple(self.labeled))
        object.__setattr__(self, 'unlabeled', tuple(self.unlabeled))
        object.__setattr__(self, 'round_budgets', tuple(self.round_budgets))
        object.__setattr__(self, 'history', tuple(self.history))

        _require_identities('pool universe', self.universe)
        _require_identities('labeled pool', self.labeled)
        _require_identities('unlabeled pool', self.unlabeled)
        _require_unique('pool universe', self.universe)
        _require_unique('labeled pool', self.labeled)
        _require_unique('unlabeled pool', self.unlabeled)
        if len(self.round_budgets) != ACTIVE_ROUNDS:
            raise ValueError('round_budgets must contain exactly five entries')
        for budget in self.round_budgets:
            _require_integer('round budget', budget)
            if budget < 0:
                raise ValueError('round budgets must not be negative')
        if sum(self.round_budgets) > len(self.universe):
            raise ValueError('total acquisition budget exceeds the target pool size')
        if len(self.history) > ACTIVE_ROUNDS:
            raise ValueError('pool history contains more than five rounds')
        if any(
            not isinstance(acquisition_round, AcquisitionRound)
            for acquisition_round in self.history
        ):
            raise TypeError('pool history must contain AcquisitionRound values')

        universe_set = set(self.universe)
        labeled_set = set(self.labeled)
        unlabeled_set = set(self.unlabeled)
        if labeled_set & unlabeled_set:
            raise ValueError('labeled and unlabeled pools must be disjoint')
        if labeled_set | unlabeled_set != universe_set:
            raise ValueError('labeled and unlabeled pools must cover the universe')

        selected_history = []
        for history_offset, acquisition_round in enumerate(self.history):
            expected_index = history_offset + 1
            if acquisition_round.round_index != expected_index:
                raise ValueError('acquisition round indices must be contiguous')
            if acquisition_round.budget != self.round_budgets[history_offset]:
                raise ValueError('acquisition history does not match the budget plan')
            selected_history.extend(acquisition_round.selected)
        _require_unique('acquisition history', selected_history)
        if tuple(selected_history) != self.labeled:
            raise ValueError('labeled pool must equal the ordered acquisition history')
        if not set(selected_history).issubset(universe_set):
            raise ValueError('acquisition history contains a sample outside the universe')

        expected_unlabeled = tuple(
            sample for sample in self.universe if sample not in labeled_set
        )
        if self.unlabeled != expected_unlabeled:
            raise ValueError('unlabeled pool must preserve the universe ordering')

    @classmethod
    def initialize(
        cls,
        samples: Iterable[SampleIdentity],
        total_budget: int,
    ) -> 'PoolState':
        universe = tuple(samples)
        return cls(
            universe=universe,
            labeled=(),
            unlabeled=universe,
            round_budgets=split_budget(total_budget, ACTIVE_ROUNDS),
        )

    @property
    def next_round_index(self) -> int:
        if self.is_complete:
            raise RuntimeError('all five acquisition rounds are already complete')
        return len(self.history) + 1

    @property
    def next_budget(self) -> int:
        if self.is_complete:
            raise RuntimeError('all five acquisition rounds are already complete')
        return self.round_budgets[len(self.history)]

    @property
    def is_complete(self) -> bool:
        return len(self.history) == ACTIVE_ROUNDS

    def acquire(self, selected: Iterable[SampleIdentity]) -> 'PoolState':
        '''Return a new state after one exact-budget acquisition round.'''

        selected_tuple = tuple(selected)
        expected_budget = self.next_budget
        if len(selected_tuple) != expected_budget:
            raise ValueError(
                'round {} requires exactly {} selected samples'.format(
                    self.next_round_index, expected_budget
                )
            )
        _require_identities('round selection', selected_tuple)
        _require_unique('round selection', selected_tuple)
        unavailable = [sample for sample in selected_tuple if sample not in self.unlabeled]
        if unavailable:
            raise ValueError(
                'selection contains an unknown or previously selected sample: {}'.format(
                    unavailable[0].qualified_id
                )
            )
        selected_set = set(selected_tuple)
        acquisition_round = AcquisitionRound(
            round_index=self.next_round_index,
            budget=expected_budget,
            selected=selected_tuple,
        )
        return PoolState(
            universe=self.universe,
            labeled=self.labeled + selected_tuple,
            unlabeled=tuple(
                sample for sample in self.unlabeled if sample not in selected_set
            ),
            round_budgets=self.round_budgets,
            history=self.history + (acquisition_round,),
        )

    def to_dict(self) -> dict:
        return {
            'schema_version': 1,
            'universe': [sample.to_dict() for sample in self.universe],
            'labeled': [sample.to_dict() for sample in self.labeled],
            'unlabeled': [sample.to_dict() for sample in self.unlabeled],
            'round_budgets': list(self.round_budgets),
            'history': [acquisition_round.to_dict() for acquisition_round in self.history],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> 'PoolState':
        if value.get('schema_version') != 1:
            raise ValueError('unsupported pool-state schema version')
        return cls(
            universe=tuple(
                SampleIdentity.from_dict(sample) for sample in value['universe']
            ),
            labeled=tuple(
                SampleIdentity.from_dict(sample) for sample in value['labeled']
            ),
            unlabeled=tuple(
                SampleIdentity.from_dict(sample) for sample in value['unlabeled']
            ),
            round_budgets=tuple(value['round_budgets']),
            history=tuple(
                AcquisitionRound.from_dict(acquisition_round)
                for acquisition_round in value['history']
            ),
        )
