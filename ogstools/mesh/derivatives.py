# SPDX-FileCopyrightText: Copyright (c) OpenGeoSys Community (opengeosys.org)
# SPDX-License-Identifier: BSD-3-Clause


import warnings
from typing import Any, Literal

import numpy as np
import pyvista as pv

DERIVATIVE_SUFFIXES = ["gradient", "divergence", "vorticity", "qcriterion"]
# NOTE: Keep in sync with ogstools.variables.variable.DERIVATIVE_SUFFIXES

from ogstools.variables.matrix import Matrix
from ogstools.variables.variable import Variable
from ogstools.variables.vector import Vector


def calculate_derivative(
    mesh: pv.UnstructuredGrid,
    variable: str | Variable,
    derivative: Literal[
        "gradient", "divergence", "vorticity", "qcriterion", "all"
    ] = "gradient",
    return_data: bool = False,
    **kwargs: Any,
) -> np.ndarray | None:
    """
    Calculate spatial gradient for specified variable.

    :param mesh:       Mesh to compute gradient on
    :param variable:   Variable of which the gradient should be computed
    :param derivative: Name of derivative to compute
    :kwargs:           kwargs for pyvista.compute_derivative
                       (scalars is replaced by variable and will
                       be ignored)
    """

    if "scalar" in kwargs:
        kwargs.pop("scalar")
        msg = """'scalar' key word argument will be ignored.\n
                    use variable key word argument instead."""
        warnings.warn(msg, UserWarning, stacklevel=3)

    if any(key in kwargs for key in DERIVATIVE_SUFFIXES):
        msg = """Defining derivative type using boolean is not supported.\n
                 Provide a string for keyword parameter derivative instead."""
        raise RuntimeError(msg)

    if isinstance(variable, str):
        variable = Variable.find(variable, mesh)

    def _pre_process(mesh: pv.UnstructuredGrid, variable: Variable) -> str:
        """
        Perform preprocessing required for computing divergence and vorticity

        :param mesh:         Mesh to preprocess.
        :param variable:     Variable to preprocess.

        :return output_name: Name of the preprocessed data set added to mesh.
        """

        output_name = variable.data_name + "_preprocessed"  # type: ignore[union-attr]

        if output_name not in mesh.array_names:
            if isinstance(variable, Vector):
                if mesh[variable.data_name].shape[1] == 2:
                    mesh[output_name] = np.pad(
                        mesh[variable.data_name],
                        ((0, 0), (0, 1)),
                        constant_values=0.0,
                    )
                elif mesh[variable.data_name].shape[1] == 3:
                    mesh[output_name] = mesh[variable.data_name]
            elif isinstance(variable, Matrix):
                mesh[output_name] = mesh[variable.data_name][:, 0:3]
            else:
                msg = """Divergence or vorticity can be calculated only\n
                            for Vector or Matrix variables."""
                raise RuntimeError(msg)

        return output_name

    derivatives = DERIVATIVE_SUFFIXES if derivative == "all" else [derivative]

    for derivative_name in derivatives:

        if derivative_name in ["divergence", "vorticity", "qcriterion"]:
            array_name = _pre_process(mesh, variable)
        elif derivative_name == "gradient":
            array_name = variable.data_name
        else:
            msg = """Incorrect derivative selected.\n
                    Only gradient, divergence and vorticity are supported"""
            raise RuntimeError(msg)

        mesh[variable.output_name] = mesh.compute_derivative(  # type: ignore[call-overload]
            scalars=array_name,
            **kwargs,
        )[
            derivative_name
        ]

    if return_data:
        return mesh[variable.output_name]
    return None


def _get_data_names(data: pv.UnstructuredGrid) -> list[str]:
    """
    Gives user list of all result data names in MeshSeries which can be
    used for plotting, post-processing, etc.
    Technical ones are ignored.
    """
    dn = (
        list(data.field_data.keys())  # type: ignore[union-attr]
        + list(data.point_data.keys())  # type: ignore[union-attr]
        + list(data.cell_data.keys())  # type: ignore[union-attr]
    )
    ignored_names = [
        "IntegrationPointMetaData",
        "OGS_VERSION",
        "_PYVISTA_USER_DICT",
        "bulk_node_ids",
    ]
    return [key for key in dn if key not in ignored_names]
