# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

from __future__ import annotations

import logging
from typing import Any

from ogstools.materiallib.schema.required_properties import (
    required_property_names,
)

from .property import MaterialProperty

logger = logging.getLogger(__name__)


class Material:
    """
    Represents a single material.

    - Can be constructed directly from YAML raw data.
    - Provides access to all properties.
    - Supports filtering by process schemas or property names.
    """

    def __init__(self, name: str, raw_data: dict[str, Any]):
        self.name = name
        self.raw = raw_data  # full YAML (e.g. for debugging or export)
        self.properties: list[MaterialProperty] = []

        self._parse_properties()

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
                prop = MaterialProperty(
                    name=prop_name, type_=type_, value=value, **extra
                )
                self.properties.append(prop)

    # -----------------------
    # Accessors
    # -----------------------
    def property_names(self) -> list[str]:
        """
        Returns a list of all property names of this material.
        """
        return [p.name for p in self.properties]

    # -----------------------
    # Filters (dummy for now)
    # -----------------------
    def filter_process(self, process_schema: dict[str, Any]) -> Material:
        """
        Return a new Material containing only properties required by a given process schema.
        """
        allowed = required_property_names(process_schema)
        return self.filter_properties(allowed)

    def filter_properties(self, allowed: set[str]) -> Material:
        """
        Return a new Material containing only the properties in 'allowed',
        preserving all extra fields (e.g. scope, unit).
        """
        filtered_props = [p for p in self.properties if p.name in allowed]
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

    # -----------------------
    # Representation
    # -----------------------
    def __repr__(self) -> str:
        return (
            f"<Material '{self.name}' with {len(self.properties)} properties>"
        )
