# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

from typing import Any


class MaterialProperty:
    def __init__(self, name: str, type_: str, value: Any = None, **extra: Any):
        # def __init__(self, name: str, type_: str, value: float | None = None, **extra):
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

    # -----------------------
    # Representation
    # -----------------------
    def __repr__(self) -> str:
        return f"• {self.name} ({self.type})"
