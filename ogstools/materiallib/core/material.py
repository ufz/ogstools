# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

from __future__ import annotations

import copy
import logging
import warnings
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import yaml

from ogstools._internal import deprecated
from ogstools.materiallib.schema.required_properties import (
    required_property_names,
)

from .property import MaterialProperty

logger = logging.getLogger(__name__)


class Material(Mapping[str, MaterialProperty]):
    """
    Represents a single material.

    - Can be constructed directly from YAML raw data.
    - Provides access to all properties.
    - Supports filtering by process schemas or property names.
    """

    __hash__ = None  # type: ignore[assignment]  # Mutable with __eq__

    def __init__(self, name: str, raw_data: dict[str, Any]):
        self.name = name
        self.raw = raw_data  # full YAML (e.g. for debugging or export)
        self.properties: list[MaterialProperty] = []
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

    def _parse_properties(self) -> None:
        block = self.raw.get("properties", {})
        if not block:
            logger.debug("Material %s has no properties", self.name)

        for prop_name, entries in block.items():
            for entry in entries if isinstance(entries, list) else [entries]:
                type_ = entry.get(
                    "type", "Constant"
                )  # Todo - Error if 'type' not found
                value = entry.get("value", None)
                extra = {
                    k: v for k, v in entry.items() if k not in ("type", "value")
                }
                self.properties.append(
                    MaterialProperty(
                        name=prop_name, type_=type_, value=value, **extra
                    )
                )

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

    def filter_process(self, process_schema: dict[str, Any]) -> Material:
        """
        Return a new Material containing only properties required by a given process schema.
        """
        allowed = required_property_names(process_schema)
        return self.filter_properties(allowed)

    def filter_properties(
        self, allowed: set[str] | str, key: str = "name"
    ) -> Material:
        """
        Return a new Material containing only the properties in 'allowed',
        preserving all extra fields (e.g. scope, unit).

        :param allowed: values to filter for
        :param key:     attribute to filter for (e.g. 'name' or 'type')
        """
        if isinstance(allowed, str):
            allowed = {allowed}

        def prop_attr(p: MaterialProperty, key: str) -> Any:
            if key in ["name", "type", "value"]:
                return getattr(p, key)
            if key not in p.extra:
                msg = f"Property {p.name} has no attribute called '{key}'."
                raise KeyError(msg)
            return p.extra[key]

        filtered_props = [
            p for p in self.properties if prop_attr(p, key) in allowed
        ]
        logger.debug(
            "Material %s: filtered %d/%d properties (%s)",
            self.name,
            len(filtered_props),
            len(self.properties),
            ", ".join(p.name for p in filtered_props),
        )

        # Build a raw_data dict with lists if multiple entries share the same name
        raw_block: dict[str, list[dict[str, Any]]] = {}
        for p in filtered_props:
            entry = {"type": p.type, "value": p.value, **p.extra}
            raw_block.setdefault(p.name, []).append(entry)

        filtered_raw = {"name": self.name, "properties": raw_block}

        # Create a new Material that parses only the filtered_raw
        return Material(name=self.name, raw_data=filtered_raw)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Material):
            return NotImplemented
        return self.name == other.name and [
            p.to_dict() for p in self.properties
        ] == [p.to_dict() for p in other.properties]

    def copy(self) -> Material:
        """Return a deep copy"""
        return copy.deepcopy(self)

    # -----------------------
    # Representation
    # -----------------------
    def __repr__(self) -> str:
        return (
            f"<Material '{self.name}' with {len(self.properties)} properties>"
        )

    def __str__(self) -> str:
        lines = [repr(self)]
        for p in self.properties:
            lines.append(f"  {p}")
        return "\n".join(lines)
