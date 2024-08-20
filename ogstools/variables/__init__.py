# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

# Author: Florian Zill (Helmholtz Centre for Environmental Research GmbH - UFZ)
"""Predefined variables for data and unit transformation."""

from functools import partial

import pandas as pd
import pyvista as pv

from . import mesh_dependent, tensor_math
from .custom_colormaps import integrity_cmap, temperature_cmap
from .matrix import Matrix
from .tensor_math import identity
from .unit_registry import u_reg
from .variable import Scalar, Variable
from .vector import Vector

__all__ = ["u_reg"]

T_MASK = "temperature_active"
H_MASK = "pressure_active"
M_MASK = "displacement_active"

# Default style to be used in plotting functions
# For now for Scalars only
# TODO: expand to Matrix and Vector
line_styles = [
    (0, ()),  # solid
    (0, (1, 1)),  # dotted
    (0, (5, 5)),  # dashed
    (0, (3, 5, 1, 5)),  # dash dotted
    (0, (3, 5, 1, 5, 1, 5)),  # dash dot dotted
    (0, (3, 10, 1, 10)),  # loosely dash dotted
]
group_color_thermal = "C6"
group_color_hydraulic = "C0"
group_color_mechanical = "C14"

# ====== general ======
material_id = Scalar(data_name="MaterialIDs", categoric=True, cmap="tab20")
# ====== thermal ======
temperature = Scalar(
    data_name="temperature",
    data_unit="K",
    output_name="temperature",
    output_unit="°C",
    symbol="T",
    mask=T_MASK,
    cmap=temperature_cmap,
    bilinear_cmap=True,
    color=group_color_thermal,
    linestyle=line_styles[0],
)
heatflowrate = Scalar(
    data_name="HeatFlowRate",
    mask=T_MASK,
    color=group_color_thermal,
    linestyle=line_styles[1],
)

# ====== hydraulic ======
pressure = Scalar(
    data_name="pressure",
    data_unit="Pa",
    output_unit="MPa",
    output_name="pore_pressure",
    symbol="p",
    mask=H_MASK,
    cmap="Blues",
    color=group_color_hydraulic,
    linestyle=line_styles[0],
)
hydraulic_head = Scalar(
    data_name="pressure",
    data_unit="m",
    output_unit="m",
    output_name="hydraulic_head",
    symbol="h",
    mask=H_MASK,
    cmap="Blues",
    color=group_color_hydraulic,
    linestyle=line_styles[1],
)
velocity = Vector(
    data_name="velocity",
    data_unit="m/s",
    output_unit="m/s",
    output_name="darcy_velocity",
    symbol="v",
    mask=H_MASK,
)
massflowrate = Scalar(data_name="MassFlowRate", mask=H_MASK)

# ====== mechanical ======
displacement = Vector(
    data_name="displacement",
    data_unit="m",
    output_unit="m",
    symbol="u",
    mask=M_MASK,
    cmap="PRGn",
    bilinear_cmap=True,
)
strain = Matrix(
    data_name="epsilon",
    data_unit="",
    output_unit="percent",
    output_name="strain",
    symbol=r"\varepsilon",
    mask=M_MASK,
)
stress = Matrix(
    data_name="sigma",
    data_unit="Pa",
    output_unit="MPa",
    output_name="stress",
    symbol=r"\sigma",
    mask=M_MASK,
)
effective_pressure = Scalar(
    data_name="sigma",
    data_unit="Pa",
    output_unit="MPa",
    output_name="effective_pressure",
    symbol=r"\pi",
    mask=M_MASK,
    func=tensor_math.effective_pressure,
    color=group_color_mechanical,
    linestyle=line_styles[0],
)
dilatancy_critescu_tot = Scalar(
    data_name="sigma",
    data_unit="Pa",
    output_unit="",
    output_name="dilatancy_criterion",
    symbol=r"F_\mathrm{dil}",
    mask=M_MASK,
    func=mesh_dependent.dilatancy_critescu,
    mesh_dependent=True,
    cmap=integrity_cmap,
    bilinear_cmap=True,
    color=group_color_mechanical,
    linestyle=line_styles[1],
)
dilatancy_critescu_eff = dilatancy_critescu_tot.replace(
    output_name="effective_dilatancy_criterion",
    func=partial(mesh_dependent.dilatancy_critescu, effective=True),
    linestyle=line_styles[2],
)

dilatancy_alkan = Scalar(
    data_name="sigma",
    data_unit="Pa",
    output_unit="MPa",
    output_name="dilatancy_criterion",
    symbol=r"F_\mathrm{dil}",
    mask=M_MASK,
    func=mesh_dependent.dilatancy_alkan,
    mesh_dependent=True,
    cmap=integrity_cmap,
    bilinear_cmap=True,
    color=group_color_mechanical,
    linestyle=line_styles[3],
)
dilatancy_alkan_eff = dilatancy_alkan.replace(
    output_name="effective_dilatancy_criterion",
    func=partial(mesh_dependent.dilatancy_alkan, effective=True),
    linestyle=line_styles[4],
)

fluid_pressure_crit = Scalar(
    data_name="sigma",
    data_unit="Pa",
    output_unit="MPa",
    output_name="fluid_pressure_criterion",
    symbol="F_p",
    mask=M_MASK,
    func=mesh_dependent.fluid_pressure_criterion,
    mesh_dependent=True,
    cmap=integrity_cmap,
    bilinear_cmap=True,
    color=group_color_mechanical,
    linestyle=line_styles[5],
)
nodal_forces = Vector(data_name="NodalForces", mask=M_MASK)

# ====== other ======
saturation = Scalar(
    data_name="Si",
    data_unit="",
    output_unit="%",
    output_name="Saturation",
    symbol="s",
)

all_variables = [v for v in locals().values() if isinstance(v, Variable)]


def get_preset(variable: Variable | str, mesh: pv.UnstructuredGrid) -> Variable:
    """
    Returns a Variable preset or creates one with correct type.

    Searches for presets by data_name and output_name and returns if found.
    If 'variable' is given as type Variable this will also look for
    derived variables (difference, aggregate).
    Otherwise create Scalar, Vector, or Matrix Variable depending on the shape
    of data in mesh.

    :param variable:    The variable to retrieve or its name if a string.
    :param mesh:        The mesh containing the variable data.
    :returns: A corresponding Variable preset or a new Variable of correct type.
    """
    data_keys: list[str] = list(set().union(mesh.point_data, mesh.cell_data))
    error_msg = (
        f"Data not found in mesh. Available data names are {data_keys}. "
    )

    if isinstance(variable, Variable):
        if variable.data_name in data_keys:
            return variable
        matches = [variable.output_name in data_key for data_key in data_keys]
        if not any(matches):
            raise KeyError(error_msg)
        data_key = data_keys[matches.index(True)]
        if data_key == f"{variable.output_name}_difference":
            return variable.difference
        return variable.replace(
            data_name=data_key,
            data_unit=variable.output_unit,
            output_unit=variable.output_unit,
            output_name=data_key,
            symbol=variable.symbol,
            func=identity,
            mesh_dependent=False,
        )

    for prop in all_variables:
        if prop.output_name == variable:
            return prop
    for prop in all_variables:
        if prop.data_name == variable:
            return prop

    matches = [variable in data_key for data_key in data_keys]
    if not any(matches):
        raise KeyError(error_msg)

    data_shape = mesh[variable].shape
    if len(data_shape) == 1:
        return Scalar(variable)
    if data_shape[1] in [2, 3]:
        return Vector(variable)
    return Matrix(variable)


def get_dataframe() -> pd.DataFrame:
    data = [
        "preset,data_name,data_unit,output_unit,output_name,type".split(",")
    ]
    for preset_name, preset_value in globals().items():
        if isinstance(preset := preset_value, Variable):
            data += [
                [
                    preset_name,
                    preset.data_name,
                    preset.data_unit,
                    preset.output_unit,
                    preset.output_name,
                    preset.type_name,
                ]
            ]

    return (
        pd.DataFrame(data[1:], columns=data[0])
        .sort_values(["data_name", "preset"])
        .set_index("preset")
    )
