# SPDX-FileCopyrightText: Copyright (c) OpenGeoSys Community (opengeosys.org)
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import logging
import re
import warnings
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, ClassVar

import yaml

from ogstools._internal import deprecated
from ogstools.materiallib.distributions import parse_distribution
from ogstools.materiallib.schema.required_properties import (
    required_property_names,
)
from ogstools.property_types import PROPERTY_TYPES

from .property import MaterialProperty, ParameterValue

logger = logging.getLogger(__name__)


class _MaterialPropertyAccessor:
    """Provide domain-based navigation of a material's flat property list.
    Bridge grouped YAML domains and the flat in-memory representation.

    :meta public:
    """

    def __init__(self, material: Material, domain: str):
        self._material = material
        self._domain = domain

    def property(self, name: str) -> MaterialProperty:
        matches = [
            prop
            for prop in self._material.properties
            if prop.name == name and prop.extra.get("domain") == self._domain
        ]
        if matches:
            return matches[0]

        available = [
            prop.name
            for prop in self._material.properties
            if prop.extra.get("domain") == self._domain
        ]
        msg = (
            f"No property with name {name} found in domain {self._domain}. "
            "Available properties are: " + ", ".join(dict.fromkeys(available))
        )
        raise KeyError(msg)


class Material(Mapping[str, MaterialProperty]):
    """
    Represents a single material.

    - Can be constructed directly from YAML raw data.
    - Provides access to all properties.
    - Supports filtering by process schemas or property names.
    """

    __hash__ = None  # type: ignore[assignment]  # Mutable with __eq__
    ALLOWED_DOMAINS: ClassVar[set[str]] = {"medium", "phase", "component"}

    def __init__(self, name: str, raw_data: dict[str, Any]):
        self.name = name
        self.raw = raw_data  # full YAML (e.g. for debugging or export)
        self.properties: list[MaterialProperty] = []
        self._validate_grouped_domains()
        self._parse_properties()

    @classmethod
    def from_file(cls, file_path: str | Path) -> Material | None:
        """Create a Material from a YAML file or return None if invalid."""
        with Path(file_path).open(encoding="utf-8") as file:
            raw_data = yaml.safe_load(file)

        if not isinstance(raw_data, dict):
            logger.debug("Skipping invalid YAML file: %s", file_path)
            return None

        name = raw_data.get("name")
        if not isinstance(name, str):
            logger.debug(
                "Skipping YAML file without valid 'name': %s", file_path
            )
            return None

        return cls(name=name, raw_data=raw_data)

    def to_file(self, file_path: str | Path) -> None:
        """Write this Material to a YAML file."""
        output_data = dict(self.raw)
        output_data["name"] = self.name
        with Path(file_path).open("w", encoding="utf-8") as file:
            yaml.safe_dump(output_data, file, sort_keys=False)

    @staticmethod
    def _validate_property_payload(
        material_name: str,
        property_name: str,
        domain_name: str,
        type_: str,
        parameters: dict[str, Any],
        metadata: dict[str, Any],
        actual_keys: set[str] | None = None,
    ) -> None:
        """Validate a material property against the shared property type registry.

        Parameters
        ----------
        material_name : str
            Name of the material.
        property_name : str
            Name of the property.
        domain_name : str
            Material domain containing the property.
        type_ : str
            OGS property type.
        parameters : dict[str, Any]
            Type-specific property parameters, e.g. "Constant", "SaturationVanGenuchten".
        metadata : dict[str, Any]
            Additional property metadata, e.g. "unit", "source".
        actual_keys : set[str] | None, optional
            Keys present in the YAML property definition. If provided, they are
            checked for missing required keys and unsupported entries.
        """
        spec = PROPERTY_TYPES.get(type_)
        if spec is None:
            msg = (
                f"Material '{material_name}' property '{property_name}' in "
                f"domain '{domain_name}' has unknown type '{type_}'."
            )
            raise ValueError(msg)

        # First validate raw YAML keys, then validate the extracted parameters.
        if actual_keys is not None:
            allowed_keys = (
                {"type"} | set(spec.parameters) | set(spec.metadata_keys)
            )
            unknown_keys = actual_keys - allowed_keys
            if unknown_keys:
                msg = (
                    f"Material '{material_name}' property '{property_name}' in "
                    f"domain '{domain_name}' of type '{type_}' contains unknown "
                    f"key(s): {', '.join(sorted(unknown_keys))}."
                )
                raise ValueError(msg)

        actual_parameter_keys = set(parameters)
        required_parameter_keys = set(spec.parameters)
        missing = required_parameter_keys - actual_parameter_keys
        unknown_parameters = actual_parameter_keys - required_parameter_keys

        if missing:
            msg = (
                f"Material '{material_name}' property '{property_name}' in "
                f"domain '{domain_name}' of type '{type_}' is missing "
                f"required parameter(s): {', '.join(sorted(missing))}."
            )
            raise ValueError(msg)

        if unknown_parameters:
            msg = (
                f"Material '{material_name}' property '{property_name}' in "
                f"domain '{domain_name}' of type '{type_}' contains unknown "
                f"parameter(s): {', '.join(sorted(unknown_parameters))}."
            )
            raise ValueError(msg)

        actual_metadata_keys = set(metadata)
        allowed_metadata_keys = set(spec.metadata_keys)
        unknown_metadata = actual_metadata_keys - allowed_metadata_keys

        if unknown_metadata:
            msg = (
                f"Material '{material_name}' property '{property_name}' in "
                f"domain '{domain_name}' of type '{type_}' contains unknown "
                f"metadata key(s): {', '.join(sorted(unknown_metadata))}."
            )
            raise ValueError(msg)

    @staticmethod
    def _parse_parameter_value(value: Any) -> ParameterValue:
        wrapper_keys = {"base_value", "distribution"}
        if not isinstance(value, Mapping):
            return ParameterValue(base_value=value)

        value_keys = set(value)
        if not (value_keys & wrapper_keys):
            return ParameterValue(base_value=value)

        unknown_keys = value_keys - wrapper_keys
        if unknown_keys:
            msg = (
                "Parameter wrapper contains unsupported key(s): "
                f"{', '.join(sorted(unknown_keys))}."
            )
            raise ValueError(msg)

        if "base_value" not in value:
            msg = "Parameter wrapper must define 'base_value'."
            raise ValueError(msg)

        distribution = value.get("distribution")
        if distribution is not None:
            if not isinstance(distribution, dict):
                msg = "Parameter wrapper key 'distribution' must be a mapping."
                raise ValueError(msg)
            parsed_distribution = parse_distribution(distribution)
        else:
            parsed_distribution = None

        return ParameterValue(
            base_value=value["base_value"], distribution=parsed_distribution
        )

    @staticmethod
    def _serialize_parameter_value(value: Any) -> Any:
        if not isinstance(value, ParameterValue):
            return value

        if value.distribution is None:
            return value.base_value

        from ogstools.materiallib.distributions import serialize_distribution

        return {
            "base_value": value.base_value,
            "distribution": serialize_distribution(value.distribution),
        }

    def _parse_properties(self) -> None:
        for domain_block in self.raw["domains"]:
            domain_name = domain_block["domain"]
            properties = domain_block["properties"]
            for prop_name, entry in properties.items():
                if "type" not in entry:
                    msg = (
                        f"Material '{self.name}' property '{prop_name}' in "
                        f"domain '{domain_name}' is missing required key "
                        "'type'."
                    )
                    raise ValueError(msg)

                type_ = entry["type"]
                spec = PROPERTY_TYPES[type_]
                parameters = {
                    k: self._parse_parameter_value(entry[k])
                    for k in spec.parameters
                }
                extra = {k: entry[k] for k in spec.metadata_keys if k in entry}

                self._validate_property_payload(
                    material_name=self.name,
                    property_name=prop_name,
                    domain_name=domain_name,
                    type_=type_,
                    parameters=parameters,
                    metadata=extra,
                    actual_keys=set(entry),
                )

                extra["domain"] = domain_name

                self.properties.append(
                    MaterialProperty(
                        name=prop_name,
                        type_=type_,
                        parameters=parameters,
                        **extra,
                    )
                )

    def _validate_grouped_domains(self) -> None:
        allowed_top_level_keys = {"name", "domains"}
        unknown_top_level_keys = set(self.raw) - allowed_top_level_keys
        if unknown_top_level_keys:
            msg = (
                f"Material '{self.name}' contains unsupported top-level "
                f"key(s): {', '.join(sorted(unknown_top_level_keys))}. "
                "Allowed keys are: domains, name."
            )
            raise ValueError(msg)

        if "properties" in self.raw:
            msg = (
                f"Material '{self.name}' must use top-level 'domains'; "
                "flat top-level 'properties' is no longer supported."
            )
            raise ValueError(msg)

        domains = self.raw.get("domains")
        if not isinstance(domains, list) or not domains:
            msg = (
                f"Material '{self.name}' must define a non-empty top-level "
                "'domains' list."
            )
            raise ValueError(msg)

        seen_domains: set[str] = set()
        for block in domains:
            if not isinstance(block, dict):
                msg = f"Material '{self.name}' has a non-mapping domain block."
                raise ValueError(msg)

            allowed_domain_keys = {"domain", "properties"}
            unknown_domain_keys = set(block) - allowed_domain_keys
            if unknown_domain_keys:
                msg = (
                    f"Material '{self.name}' contains unsupported key(s) in "
                    f"a domain block: {', '.join(sorted(unknown_domain_keys))}. "
                    "Allowed keys are: domain, properties."
                )
                raise ValueError(msg)

            domain_name = block.get("domain")
            if not isinstance(domain_name, str):
                msg = (
                    f"Material '{self.name}' has a domain block without a "
                    "valid 'domain' string."
                )
                raise ValueError(msg)
            if domain_name not in self.ALLOWED_DOMAINS:
                msg = (
                    f"Material '{self.name}' uses unsupported domain "
                    f"'{domain_name}'. Allowed domains are: "
                    f"{sorted(self.ALLOWED_DOMAINS)}."
                )
                raise ValueError(msg)
            if domain_name in seen_domains:
                msg = (
                    f"Material '{self.name}' defines duplicate top-level "
                    f"domain block '{domain_name}'."
                )
                raise ValueError(msg)
            seen_domains.add(domain_name)

            properties = block.get("properties")
            if not isinstance(properties, dict):
                msg = (
                    f"Material '{self.name}' domain '{domain_name}' must "
                    "define a 'properties' mapping."
                )
                raise ValueError(msg)

            for prop_name, prop_entry in properties.items():
                if not isinstance(prop_entry, dict):
                    msg = (
                        f"Material '{self.name}' property '{prop_name}' in domain "
                        f"'{domain_name}' must be a mapping, not {type(prop_entry).__name__}."
                    )
                    raise ValueError(msg)

    def __getitem__(self, key: str) -> MaterialProperty:
        for p in self.properties:
            if p.name == key:
                return p
        msg = (
            f"No property with name {key} found. Available properties are: "
            + ", ".join(self)
        )
        raise KeyError(msg)

    def __iter__(self) -> Iterator[str]:
        return iter(dict.fromkeys(p.name for p in self.properties))

    def __len__(self) -> int:
        return len(self.properties)

    def __bool__(self) -> bool:
        return bool(self.name)

    @property
    def medium(self) -> _MaterialPropertyAccessor:
        return _MaterialPropertyAccessor(self, "medium")

    @property
    def phase(self) -> _MaterialPropertyAccessor:
        return _MaterialPropertyAccessor(self, "phase")

    @property
    def component(self) -> _MaterialPropertyAccessor:
        return _MaterialPropertyAccessor(self, "component")

    @staticmethod
    def _raw_from_properties(
        name: str, properties: list[MaterialProperty]
    ) -> dict:
        "Return grouped raw YAML data without list-valued properties."
        domain_blocks: dict[str, dict[str, dict[str, Any]]] = {}

        for p in properties:
            domain = p.extra.get("domain")
            if not isinstance(domain, str):
                msg = (
                    f"Property '{p.name}' in material '{name}' is missing its "
                    "domain metadata."
                )
                raise ValueError(msg)

            metadata = {k: v for k, v in p.extra.items() if k != "domain"}
            Material._validate_property_payload(
                material_name=name,
                property_name=p.name,
                domain_name=domain,
                type_=p.type,
                parameters=p.parameters,
                metadata=metadata,
            )

            entry = {
                "type": p.type,
                **{
                    key: Material._serialize_parameter_value(value)
                    for key, value in p.parameters.items()
                },
                **metadata,
            }

            properties_by_name = domain_blocks.setdefault(domain, {})
            if p.name in properties_by_name:
                msg = (
                    f"Material '{name}' contains duplicate property '{p.name}' "
                    f"in domain '{domain}', which can no longer be exported."
                )
                raise ValueError(msg)

            properties_by_name[p.name] = entry

        return {
            "name": name,
            "domains": [
                {"domain": domain, "properties": properties_by_name}
                for domain, properties_by_name in domain_blocks.items()
            ],
        }

    @property
    def property_names(self) -> list[str]:
        """Returns a list of all property names of this material."""
        return list(self)

    @deprecated(""": use mat[key] instead.""")
    def get_property(self, key: str) -> MaterialProperty:
        warnings.warn(
            "get_property() is deprecated, use mat[key] instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self[key]

    def filter_process(self, process_schema: dict[str, Any]) -> None:
        """
        Filter self, to only contain properties required by a given process.
        """
        allowed = required_property_names(process_schema)
        self.filter_properties(allowed)

    def filter_properties(
        self, allowed: set[str] | str, key: str = "name"
    ) -> None:
        """
        Filter self, to only contain properties in 'allowed',
        preserving all extra fields (e.g. source, unit).

        :param allowed: values to filter for
        :param key:     attribute to filter for (e.g. 'name' or 'type')
        """
        if isinstance(allowed, str):
            allowed = {allowed}

        filtered_props = [p for p in self.properties if p.get(key) in allowed]
        logger.debug(
            "Material %s: filtered %d/%d properties (%s)",
            self.name,
            len(filtered_props),
            len(self.properties),
            ", ".join(p.name for p in filtered_props),
        )

        self.properties = filtered_props
        self.raw = Material._raw_from_properties(self.name, filtered_props)

    @property
    def duplicates(self) -> list[MaterialProperty]:
        "Returns all material properties with multiple definitions."
        prop_names = [p.name for p in self.properties]
        dupe_names = [x for x in self.property_names if prop_names.count(x) > 1]
        return [p for p in self.properties if p.name in dupe_names]

    def _filter(
        self, selection: dict[str, dict[str, str | re.Pattern]]
    ) -> None:
        "Reduce properties by the given selection."
        if len(selection) == 0:
            return

        def matching(value: str | re.Pattern, candidate: Any) -> bool:
            if isinstance(value, re.Pattern):
                return (
                    isinstance(candidate, str)
                    and re.search(value, candidate) is not None
                )
            return candidate == value

        pick: list[MaterialProperty] = []
        for name, restrictions in selection.items():
            filtered = [
                p
                for p in self.properties
                if matching(name, p.name)
                and all(matching(v, p.get(k)) for k, v in restrictions.items())
            ]
            pick += filtered

        others = [
            p
            for p in self.properties
            if not any(matching(name, p.name) for name in selection)
        ]

        self.properties = sorted(others + pick, key=lambda p: p.name)
        self.raw = Material._raw_from_properties(self.name, self.properties)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Material):
            return NotImplemented

        def sort_key(d: dict) -> list:
            return sorted((k, str(v)) for k, v in d.items())

        return self.name == other.name and sorted(
            [p.to_dict() for p in self.properties], key=sort_key
        ) == sorted([p.to_dict() for p in other.properties], key=sort_key)

    def copy(
        self, selection: dict[str, dict[str, str | re.Pattern]] | None = None
    ) -> Material:
        """Return a deep copy, optionally with a filtered selection.

        :param selection:
            Maps restrictions to different properties. They will be only present
            in the resulting copy, if the properties named in selection adhere
            to the given constraint. The values can be regular expressions.
            Shape: `{"propertynames": {"attributes": "values"}}`

            Example:

            `{"saturation": {"type": re.compile("SaturationVan.*")},
            "density": {"type": "Constant", "source": re.compile(".*2018.*")}}`
        """
        new_mat = copy.deepcopy(self)
        if selection is not None:
            new_mat._filter(selection)
        return new_mat

    def __repr__(self) -> str:
        return (
            f"<Material '{self.name}' with {len(self.properties)} properties>"
        )

    def __str__(self) -> str:
        lines = [repr(self)]
        for p in self.properties:
            lines.append(f"  {p}")
        return "\n".join(lines)
