# SPDX-FileCopyrightText: Copyright (c) OpenGeoSys Community (opengeosys.org)
# SPDX-License-Identifier: BSD-3-Clause

"""Class to store functions and arguments for Variables."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

T = TypeVar("T")

if TYPE_CHECKING:
    from .variable import Variable


def identity(vals: T) -> T:
    "Return the input values."
    return vals


@dataclass
class Function:
    """Store the function, it's arguments and parameters for later usage.

    :param callable:    The callable function.
    :param args:        The arguments of the function in correct order
    :param params:      The parameters as a dict of parameter names and values.
    :param aggregating: A boolean flag indicating if the function is aggregating
                        the underylying data.
    """

    callable: Callable = identity
    args: list[str | Variable | Callable] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    aggregating: bool = False
