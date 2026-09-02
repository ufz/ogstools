# SPDX-FileCopyrightText: Copyright (c) OpenGeoSys Community (opengeosys.org)
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Any

from ogstools.materiallib.distributions import Distribution


@dataclass(frozen=True)
class ParameterValue:
    base_value: Any
    distribution: Distribution | None = None


class MaterialProperty:
    def __init__(
        self,
        name: str,
        type_: str,
        parameters: dict[str, ParameterValue],
        **extra: Any,
    ):
        self.name = name
        self.type = type_
        self.parameters = parameters
        self.extra = extra  # e.g. unit, slope, source, ...

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name, "type": self.type}
        d.update(self.parameters)
        d.update(self.extra)
        return d

    def __repr__(self) -> str:
        return f"{self.name} ({self.type})"

    def __str__(self) -> str:
        lines = [f"{self.name} ({self.type})"]
        for k, v in self.parameters.items():
            lines.append(f"  {k}: {v}")
        for k, v in self.extra.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)

    def parameter(self, name: str) -> ParameterValue:
        if name not in self.parameters:
            msg = f"Property {self.name} has no parameter called '{name}'."
            raise KeyError(msg)
        return self.parameters[name]

    def get(self, key: str, default: str | None = None) -> Any:
        if key in ["name", "type", "parameters"]:
            return getattr(self, key)
        if key in self.parameters:
            return self.parameters[key]
        if key not in self.extra:
            if default is None:
                msg = f"Property {self.name} has no attribute called '{key}'."
                raise KeyError(msg)
            return default
        return self.extra[key]
