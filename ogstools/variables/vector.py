# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from pint.facets.plain import PlainQuantity

from ogstools.variables.variable import Scalar, Variable

from .tensor_math import _split_quantity, _to_quantity

ValType: TypeAlias = PlainQuantity | np.ndarray


def vector_norm(values: ValType) -> ValType:
    ":returns: The norm of the vector."
    vals, unit = _split_quantity(values)
    result = np.linalg.norm(vals, axis=-1)
    return _to_quantity(result, unit)


@dataclass
class Vector(Variable):
    """Represent a vector variable.

    Vector variables should contain either 2 (2D) or 3 (3D) components.
    Vector components can be accesses with brackets e.g. displacement[0]
    """

    def __getitem__(self, index: int | Literal["x", "y", "z"]) -> Scalar:
        """
        Get a scalar variable as a specific component of the vector variable.

        :param index: The index of the component.

        :returns: A scalar variable as a vector component.
        """
        int_index = index if isinstance(index, int) else "xyz".index(index)
        return Scalar.from_variable(
            self,
            output_name=self.output_name + f"_{index}",
            symbol=f"{{{self.symbol}}}_{index}",
            func=lambda x: self.func(x)[..., int_index],
            bilinear_cmap=True,
        )

    @property
    def magnitude(self) -> Scalar:
        ":returns: A scalar variable as the magnitude of the vector."
        return Scalar.from_variable(
            self,
            output_name=self.output_name + "_magnitude",
            symbol=f"||{{{self.symbol}}}||",
            func=lambda x: vector_norm(self.func(x)),
        )


@dataclass
class VectorList(Variable):
    """Represent a list of vector variables."""

    def __getitem__(self, index: int) -> Vector:
        ":returns: A vector variable as a component of the vectorlist variable."
        return Vector.from_variable(
            self,
            output_name=self.output_name + f"_{index}",
            symbol=f"{{{self.symbol}}}_{index}",
            func=lambda x: np.take(self.func(x), index, axis=-1),
        )
