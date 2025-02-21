# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, Literal, TypeAlias, TypeVar

import numpy as np
from pint.facets.plain import PlainQuantity

from ogstools.variables.variable import Scalar, Variable

from .tensor_math import _split_quantity, _to_quantity

ValType: TypeAlias = PlainQuantity | np.ndarray

T = TypeVar("T")


def vector_norm(values: ValType) -> ValType:
    ":returns: The norm of the vector."
    vals, unit = _split_quantity(values)
    result = np.linalg.norm(vals, axis=-1)
    return _to_quantity(result, unit)


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
class BHE_Vector(Variable):
    """
    ========= ===========================
    BHE type  available Vector components
    ========= ===========================
    1U        in, out, grout1, grout2
    2U        in1, in2, out1, out2, grout1, grout2, grout3, grout4
    1P        in, grout
    CXC       in, out, grout
    CXA       in, out, grout
    ========= ===========================
    """

    BHE_COMPONENTS: ClassVar[dict[str, list[str]]] = {
        "1U":  ["in", "out", "grout1", "grout2"],
        "2U":  ["in1", "in2", "out1", "out2", "grout1", "grout2", "grout3", "grout4"],
        "CXC": ["in", "out", "grout"],
        "CXA": ["in", "out", "grout"],
        "1P":  ["in", "grout"],
    }  # fmt: skip

    def __getitem__(self, index: int | str | tuple) -> Scalar:
        """
        Get a scalar variable as a specific component of the vector variable.

        :param index: The index of the component.

        :returns: A scalar variable as a vector component.
        """

        if isinstance(index, tuple) and len(index) > 2:
            msg = "Expected at most two indices: (BHE number, component)"
            raise IndexError(msg)
        suffix = f"{index[0]}" if isinstance(index, tuple) else ""
        comp_index = index[1] if isinstance(index, tuple) else index

        def get_component(
            comp_index: int | str | list[int] | list[str],
        ) -> Callable:

            def component_selector(x: T) -> T:
                data: np.ndarray = self.func(x)
                len_data = data.shape[-1]

                for _, components in BHE_Vector.BHE_COMPONENTS.items():
                    if len_data == len(components):
                        if isinstance(comp_index, list):
                            if all(isinstance(i, int) for i in comp_index):
                                return data[..., comp_index]
                            if all(isinstance(i, str) for i in comp_index):
                                component_index_list = []
                                for comp in comp_index:
                                    assert isinstance(
                                        comp, str
                                    )  # Type assertion to make mypy happy
                                    component_index_list.append(
                                        components.index(comp)
                                    )
                                return data[..., component_index_list]

                            msg = f"Unknown str index list {comp_index}"
                            raise ValueError(msg)
                        if isinstance(comp_index, str):
                            if comp_index in components:
                                return data[..., components.index(comp_index)]
                            msg = f"Unknown str index {comp_index}"
                            raise ValueError(msg)
                        if isinstance(comp_index, int):
                            return data[..., comp_index]
                msg = f"Unknown BHE type with BHE vector length {len_data}"
                raise ValueError(msg)

            return component_selector

        return Scalar.from_variable(
            self,
            data_name=self.data_name + suffix,
            output_name=self.output_name + suffix + f"_{comp_index}",
            symbol=f"{{{self.symbol}}}_{comp_index}",
            func=get_component(comp_index),
        )

    @property
    def magnitude(self) -> Scalar:
        ":returns: A scalar variable as the magnitude of the vector."
        msg = """You tried to get the magnitude of a BHE temperature vector,
        which most likely is unintended. Please access the different components
        via indexing: e.g. ot.variables.temperature_BHE["T_in"].\n""" + str(
            BHE_Vector.__doc__
        )
        raise TypeError(msg)


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
