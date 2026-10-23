# SPDX-FileCopyrightText: Copyright (c) OpenGeoSys Community (opengeosys.org)
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, get_args

import numpy as np

from ogstools.variables.matrix import Matrix
from ogstools.variables.variable import DERIVATIVE_SUFFIXES, Variable
from ogstools.variables.vector import Vector

if TYPE_CHECKING:
    from pyvista import UnstructuredGrid


def calculate_derivative(
    mesh: UnstructuredGrid, variable: str | Variable, **kwargs: Any
) -> np.ndarray:
    """
    Calculate the variable derivative for a given mesh.

    :param mesh:       Mesh to compute the derivative on
    :param variable:   Variable of which the derivative should be computed
    :kwargs:           kwargs for pyvista.compute_derivative
                       (scalars is replaced by variable and will
                       be ignored)
    """
    if "scalars" in kwargs:
        kwargs.pop("scalars")
        msg = """'scalars' key word argument will be ignored.
                    use variable key word argument instead."""
        warnings.warn(msg, UserWarning, stacklevel=3)

    if any(key in kwargs for key in get_args(DERIVATIVE_SUFFIXES)):
        msg = """Defining derivative type using boolean is not supported.
                 Use variable methods instead, e.g.
                 `ot.variables.temperature.gradient`."""
        raise TypeError(msg)
    variable = Variable.find(variable, mesh)

    derivative = variable.output_name.rsplit("_", 1)[-1]
    if derivative not in get_args(DERIVATIVE_SUFFIXES):
        msg = (
            "Can only be called with a derivative variable."
            f"Given suffix is {derivative}."
            f"Possible are: {get_args(DERIVATIVE_SUFFIXES)}."
        )
        raise ValueError(msg)
    base_var = variable.parent
    if isinstance(base_var, Vector) and derivative == "gradient":
        msg = (
            "Gradients of vectors are currently not supported yet. Please "
            "call gradient on a component of the vector."
        )
        raise TypeError(msg)

    if isinstance(base_var, Matrix):
        result = []
        for idx in range(3):
            col_var = base_var.col(idx).replace(output_unit=base_var.data_unit)
            mesh[col_var.output_name] = col_var.transform(mesh)
            result.append(
                calculate_derivative(mesh, getattr(col_var, derivative))
            )
        return np.moveaxis(result, 0, 1)

    def _pre_process(mesh: UnstructuredGrid, variable: Variable) -> str:
        """
        Perform preprocessing required for computing divergence and vorticity

        :param mesh:         Mesh to preprocess.
        :param variable:     Variable to preprocess.

        :return output_name: Name of the preprocessed data set added to mesh.
        """

        output_name = f"{variable.output_name}_preprocessed"

        if output_name not in mesh.array_names:
            if isinstance(variable, Vector):
                num_comps = mesh[variable.output_name].shape[1]
                if num_comps == 2:
                    mesh[output_name] = np.pad(
                        mesh[variable.output_name],
                        ((0, 0), (0, 1)),
                        constant_values=0.0,
                    )
                elif num_comps == 3:
                    mesh[output_name] = mesh[variable.output_name]
                else:
                    msg = (
                        f"{variable.output_name} has an unsupported number "
                        f"of components {num_comps}."
                    )
                    raise TypeError(msg)
            else:
                msg = """Divergence or vorticity can be calculated only\n
                            for Vector or Matrix variables."""
                raise TypeError(msg)

        return output_name

    if base_var.output_name not in mesh.array_names:
        if base_var.process_with_units:
            mesh[base_var.output_name] = base_var.transform(mesh)
        else:
            mesh[base_var.output_name] = base_var.replace(
                output_unit=base_var.data_unit
            ).transform(mesh)
    if derivative in ["divergence", "vorticity", "qcriterion"]:
        array_name = _pre_process(mesh, base_var)
    else:  # if derivative == "gradient":
        array_name = base_var.output_name

    kwargs[derivative] = True
    kwargs.setdefault("gradient", False)
    return mesh.compute_derivative(scalars=array_name, **kwargs)[
        derivative
    ]  # pyright: ignore[reportReturnType, reportIndexIssue]
