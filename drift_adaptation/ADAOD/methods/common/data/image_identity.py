'''Stable, namespaced identities for samples in ADAOD pools.'''

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping


_NAMESPACE_PATTERN = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]*$')


@dataclass(frozen=True, order=True)
class SampleIdentity:
    '''A dataset-scoped sample identifier.

    ``sample_id`` may be a relative dataset path. The namespace is deliberately
    more restrictive so that the qualified representation remains unambiguous.
    '''

    namespace: str
    sample_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.namespace, str) or not _NAMESPACE_PATTERN.fullmatch(
            self.namespace
        ):
            raise ValueError(
                'namespace must start with an alphanumeric character and contain '
                'only alphanumerics, dots, underscores, or hyphens'
            )
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise ValueError('sample_id must be a non-empty string')
        if self.sample_id != self.sample_id.strip():
            raise ValueError('sample_id must not contain leading or trailing whitespace')
        if any(ord(character) < 32 for character in self.sample_id):
            raise ValueError('sample_id must not contain control characters')

    @property
    def qualified_id(self) -> str:
        '''Return the stable ``namespace:sample_id`` representation.'''

        return '{}:{}'.format(self.namespace, self.sample_id)

    def to_dict(self) -> dict:
        return {'namespace': self.namespace, 'sample_id': self.sample_id}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> 'SampleIdentity':
        if set(value) != {'namespace', 'sample_id'}:
            raise ValueError('sample identity must contain namespace and sample_id')
        return cls(namespace=value['namespace'], sample_id=value['sample_id'])

    @classmethod
    def parse(cls, qualified_id: str) -> 'SampleIdentity':
        if not isinstance(qualified_id, str) or ':' not in qualified_id:
            raise ValueError('qualified sample identity must contain a namespace separator')
        namespace, sample_id = qualified_id.split(':', 1)
        return cls(namespace=namespace, sample_id=sample_id)
