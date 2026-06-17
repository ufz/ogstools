# SPDX-FileCopyrightText: Copyright (c) OpenGeoSys Community (opengeosys.org)
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PropertyAddress:
    domain: str
    property_name: str
    index: int = 0
    parameter_path: tuple[str, ...] = ()


class MaterialProperty:
    def __init__(self, name: str, type_: str, value: Any = None, **extra: Any):
        self.name = name
        self.type = type_
        self.value = value
        self.extra = extra  # e.g. unit, slope, source, ...

    @staticmethod
    def _is_scalar_metadata_wrapper(value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        keys = set(value)
        if "value" not in keys:
            return False
        return keys.issubset({"value", "unit", "distribution"})

    def to_dict(self) -> dict:
        d = {"name": self.name, "type": self.type}
        if self.value is not None:
            d["value"] = self.value
        d.update(self.extra)
        return d

    def __repr__(self) -> str:
        return f"{self.name} ({self.type})"

    def __str__(self) -> str:
        lines = [f"{self.name} ({self.type})", f"  value: {self.value}"]
        for k, v in self.extra.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)

    def get(self, key: str, default: str | None = None) -> Any:
        if key in ["name", "type", "value"]:
            return getattr(self, key)
        if key not in self.extra:
            if default is None:
                msg = f"Property {self.name} has no attribute called '{key}'."
                raise KeyError(msg)
            return default
        return self.extra[key]
