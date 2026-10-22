# SPDX-FileCopyrightText: Copyright (c) OpenGeoSys Community (opengeosys.org)
# SPDX-License-Identifier: BSD-3-Clause

"""Class to store functions and arguments for Variables."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .tensor_math import identity

if TYPE_CHECKING:
    from .variable import Variable


@dataclass
class Function:
    callable: Callable = identity
    args: list[str | Variable | Callable] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    aggregating: bool = False
