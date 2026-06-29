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
from ogstools.materiallib.schema.required_properties import (
    required_property_names,
)

from .property import MaterialProperty, PropertyAddress

logger = logging.getLogger(__name__)


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
    def from_file(cls, file_path: str | Path) -> Material:
        """Create a Material from a YAML file."""
        with Path(file_path).open(encoding="utf-8") as file:
            raw_data = yaml.safe_load(file)

        if not isinstance(raw_data, dict):
            msg = f"Material file '{file_path}' must contain a YAML mapping."
            raise ValueError(msg)

        name = raw_data.get("name")
        if not isinstance(name, str):
            msg = (
                f"Material file '{file_path}' must define a top-level "
                "'name' string."
            )
            raise ValueError(msg)

        return cls(name=name, raw_data=raw_data)

    def to_file(self, file_path: str | Path) -> None:
        """Write this Material to a YAML file."""
        output_data = dict(self.raw)
        output_data["name"] = self.name
        with Path(file_path).open("w", encoding="utf-8") as file:
            yaml.safe_dump(output_data, file, sort_keys=False)

    def _parse_properties(self) -> None:
        for domain_block in self.raw["domains"]:
            domain_name = domain_block["domain"]
            properties = domain_block["properties"]
            for prop_name, entries in properties.items():
                for entry in (
                    entries if isinstance(entries, list) else [entries]
                ):
                    type_ = entry.get(
                        "type", "Constant"
                    )  # TODO: Error if 'type' not found
                    raw_value = entry.get("value", None)
                    parsed_value, value_extra = self._parse_top_level_value(
                        raw_value
                    )
                    extra = {
                        k: v
                        for k, v in entry.items()
                        if k not in ("type", "value")
                    }
                    extra.update(value_extra)
                    extra["domain"] = domain_name
                    self.properties.append(
                        MaterialProperty(
                            name=prop_name,
                            type_=type_,
                            value=parsed_value,
                            **extra,
                        )
                    )

    @staticmethod
    def _is_scalar_metadata_wrapper(value: Any) -> bool:
        return MaterialProperty._is_scalar_metadata_wrapper(value)

    @classmethod
    def _parse_top_level_value(cls, value: Any) -> tuple[Any, dict[str, Any]]:
        if not cls._is_scalar_metadata_wrapper(value):
            return value, {}

        wrapped_value = dict(value)
        nominal_value = wrapped_value.pop("nominal_value")
        return nominal_value, wrapped_value

    def _property_by_address(
        self, address: PropertyAddress
    ) -> MaterialProperty:
        matches = [
            prop
            for prop in self.properties
            if prop.name == address.property_name
            and prop.extra.get("domain") == address.domain
        ]
        if address.index < 0 or address.index >= len(matches):
            msg = (
                f"No property found for address {address!r} in material "
                f"'{self.name}'."
            )
            raise IndexError(msg)
        return matches[address.index]

    def _parameter_payload(self, address: PropertyAddress) -> Any:
        prop = self._property_by_address(address)
        if not address.parameter_name:
            return prop.value

        current: Any = prop.extra
        segment = address.parameter_name
        if not isinstance(current, Mapping) or segment not in current:
            msg = (
                f"Parameter name {address.parameter_name!r} does not exist "
                f"for address {address!r} in material '{self.name}'."
            )
            raise KeyError(msg)
        return current[segment]

    def nominal_value(self, address: PropertyAddress) -> Any:
        payload = self._parameter_payload(address)
        if self._is_scalar_metadata_wrapper(payload):
            return payload["nominal_value"]
        return payload

    def distribution(self, address: PropertyAddress) -> dict[str, Any] | None:
        if not address.parameter_name:
            prop = self._property_by_address(address)
            authored_distribution = prop.extra.get("distribution")
            return (
                authored_distribution
                if isinstance(authored_distribution, dict)
                else None
            )

        payload = self._parameter_payload(address)
        if self._is_scalar_metadata_wrapper(payload):
            authored_distribution = payload.get("distribution")
            return (
                authored_distribution
                if isinstance(authored_distribution, dict)
                else None
            )
        return None

    def _validate_grouped_domains(self) -> None:
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

    @staticmethod
    def _raw_from_properties(
        name: str, properties: list[MaterialProperty]
    ) -> dict:
        "raw grouped yaml data dict with lists if multiple entries have same names"
        domain_blocks: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for p in properties:
            domain = p.extra.get("domain")
            if not isinstance(domain, str):
                msg = (
                    f"Property '{p.name}' in material '{name}' is missing its "
                    "domain metadata."
                )
                raise ValueError(msg)
            entry_extra = {k: v for k, v in p.extra.items() if k != "domain"}
            value = p.value
            if "distribution" in entry_extra:
                wrapped_value = {"value": value}
                if "unit" in entry_extra:
                    wrapped_value["unit"] = entry_extra.pop("unit")
                wrapped_value["distribution"] = entry_extra.pop("distribution")
                value = wrapped_value

            entry = {
                "type": p.type,
                "value": value,
                **entry_extra,
            }
            properties_by_name = domain_blocks.setdefault(domain, {})
            properties_by_name.setdefault(p.name, []).append(entry)

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
        preserving all extra fields (e.g. scope, unit).

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

        def matching(value: str | re.Pattern, text: str) -> bool:
            if isinstance(value, re.Pattern):
                return re.search(value, text) is not None
            return text == value

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
