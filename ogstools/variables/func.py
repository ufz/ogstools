# SPDX-FileCopyrightText: Copyright (c) OpenGeoSys Community (opengeosys.org)
# SPDX-License-Identifier: BSD-3-Clause

"""Class to store functions and arguments for Variables."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .tensor_math import identity


@dataclass
class Function:
    callable: Callable = identity
    args: list[str | Callable] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
