import pytest

from methods.common.data.image_identity import SampleIdentity


def test_identity_is_namespaced_and_round_trips() -> None:
    city = SampleIdentity('cityscapes', 'frankfurt/frame:0001')
    foggy = SampleIdentity('foggy-cityscapes', 'frankfurt/frame:0001')

    assert city != foggy
    assert city.qualified_id == 'cityscapes:frankfurt/frame:0001'
    assert SampleIdentity.parse(city.qualified_id) == city
    assert SampleIdentity.from_dict(city.to_dict()) == city


@pytest.mark.parametrize(
    ('namespace', 'sample_id'),
    [
        ('', 'frame'),
        ('foggy cityscapes', 'frame'),
        (':cityscapes', 'frame'),
        ('cityscapes', ''),
        ('cityscapes', ' frame'),
        ('cityscapes', 'frame\n'),
    ],
)
def test_identity_rejects_ambiguous_values(namespace: str, sample_id: str) -> None:
    with pytest.raises(ValueError):
        SampleIdentity(namespace, sample_id)


def test_parse_requires_namespace_separator() -> None:
    with pytest.raises(ValueError, match='separator'):
        SampleIdentity.parse('frame-without-namespace')
