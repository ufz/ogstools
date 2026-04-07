# SPDX-FileCopyrightText: Copyright (c) OpenGeoSys Community (opengeosys.org)
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import logging
import re
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
                )  # TODO: Error if 'type' not found
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

    @staticmethod
    def _raw_from_properties(
        name: str, properties: list[MaterialProperty]
    ) -> dict:
        "raw yaml data dict with lists if multiple entries have same names"
        raw_block: dict[str, list[dict[str, Any]]] = {}
        for p in properties:
            entry = {"type": p.type, "value": p.value, **p.extra}
            raw_block.setdefault(p.name, []).append(entry)

        return {"name": name, "properties": raw_block}

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
            if not all(matching(name, p.name) for name in selection)
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
