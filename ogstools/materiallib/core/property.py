# SPDX-FileCopyrightText: Copyright (c) OpenGeoSys Community (opengeosys.org)
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any


class MaterialProperty:
    def __init__(self, name: str, type_: str, value: Any = None, **extra: Any):
        self.name = name
        self.type = type_
        self.value = value
        self.extra = extra  # e.g. unit, slope, source, ...

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
