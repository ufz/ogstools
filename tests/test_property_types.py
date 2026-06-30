from lxml import etree as ET

from ogstools.ogs6py.media import Media
from ogstools.property_types import PROPERTY_TYPES, PropertyTypeSpec


def test_property_type_spec_defaults() -> None:
    spec = PropertyTypeSpec(parameters=("value",))

    assert spec.parameters == ("value",)
    assert spec.metadata_keys == ("unit", "source")


def test_property_types_registry_contains_expected_examples() -> None:
    assert PROPERTY_TYPES["Constant"].parameters == ("value",)
    assert PROPERTY_TYPES["SaturationVanGenuchten"].parameters == (
        "exponent",
        "p_b",
        "residual_gas_saturation",
        "residual_liquid_saturation",
    )


def test_property_types_have_disjoint_parameter_and_metadata_names() -> None:
    for name, spec in PROPERTY_TYPES.items():
        assert set(spec.parameters).isdisjoint(spec.metadata_keys), name


def test_media_uses_property_type_registry_for_all_entries() -> None:
    media = Media(ET.ElementTree(ET.Element("OpenGeoSysProject")))
    for name, spec in PROPERTY_TYPES.items():
        assert name in media.properties
        assert media.properties[name] == list(spec.parameters)
