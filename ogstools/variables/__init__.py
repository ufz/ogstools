# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

# Author: Florian Zill (Helmholtz Centre for Environmental Research GmbH - UFZ)
"""Predefined variables for data and unit transformation."""

from functools import partial

import numpy as np
import pandas as pd
import pyvista as pv

from . import mesh_dependent, tensor_math
from .custom_colormaps import integrity_cmap, temperature_cmap
from .matrix import Matrix
from .unit_registry import u_reg
from .variable import Scalar, Variable
from .vector import BHE_Vector, Vector

__all__ = ["u_reg"]

T_MASK = "temperature_active"
H_MASK = "pressure_active"
M_MASK = "displacement_active"

# Default colors to be used in plotting functions
COLOR_THERMAL = "tab:red"
COLOR_HYDRO = "tab:blue"
COLOR_MECH = "black"  # green would be bad for colorblindess

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
    color=COLOR_THERMAL,
)
heatflowrate = Scalar(
    data_name="HeatFlowRate", mask=T_MASK, color=COLOR_THERMAL
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
    color=COLOR_HYDRO,
)
hydraulic_head = Scalar(
    data_name="pressure",
    data_unit="m",
    output_unit="m",
    output_name="hydraulic_head",
    symbol="h",
    mask=H_MASK,
    cmap="Blues",
    color=COLOR_HYDRO,
)
velocity = Vector(
    data_name="velocity",
    data_unit="m/s",
    output_unit="m/s",
    output_name="darcy_velocity",
    symbol="v",
    mask=H_MASK,
    cmap="Blues",
    color=COLOR_HYDRO,
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
    color=COLOR_MECH,
    bilinear_cmap=True,
)
strain = Matrix(
    data_name="epsilon",
    data_unit="",
    output_unit="percent",
    output_name="strain",
    symbol=r"\varepsilon",
    color=COLOR_MECH,
    mask=M_MASK,
)
stress = Matrix(
    data_name="sigma",
    data_unit="Pa",
    output_unit="MPa",
    output_name="stress",
    symbol=r"\sigma",
    color=COLOR_MECH,
    mask=M_MASK,
)
effective_pressure = Scalar(
    data_name="sigma",
    data_unit="Pa",
    output_unit="MPa",
    output_name="effective_pressure",
    symbol=r"\pi",
    func=tensor_math.effective_pressure,
    mask=M_MASK,
    color=COLOR_MECH,
)
dilatancy_critescu_tot = Scalar(
    data_name="sigma",
    data_unit="Pa",
    output_unit="",
    output_name="dilatancy_criterion",
    symbol=r"F_\mathrm{dil}",
    func=mesh_dependent.dilatancy_critescu,
    mask=M_MASK,
    color=COLOR_MECH,
    mesh_dependent=True,
    cmap=integrity_cmap,
    bilinear_cmap=True,
)
dilatancy_critescu_eff = dilatancy_critescu_tot.replace(
    output_name="effective_dilatancy_criterion",
    func=partial(mesh_dependent.dilatancy_critescu, effective=True),
)

dilatancy_alkan = Scalar(
    data_name="sigma",
    data_unit="Pa",
    output_unit="MPa",
    output_name="dilatancy_criterion",
    symbol=r"F_\mathrm{dil}",
    func=mesh_dependent.dilatancy_alkan,
    mask=M_MASK,
    color=COLOR_MECH,
    mesh_dependent=True,
    cmap=integrity_cmap,
    bilinear_cmap=True,
)
dilatancy_alkan_eff = dilatancy_alkan.replace(
    output_name="effective_dilatancy_criterion",
    func=partial(mesh_dependent.dilatancy_alkan, effective=True),
)

fluid_pressure_crit = Scalar(
    data_name="sigma",
    data_unit="Pa",
    output_unit="MPa",
    output_name="fluid_pressure_criterion",
    symbol="F_p",
    func=mesh_dependent.fluid_pressure_criterion,
    mask=M_MASK,
    color=COLOR_MECH,
    mesh_dependent=True,
    cmap=integrity_cmap,
    bilinear_cmap=True,
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

temperature_BHE = BHE_Vector(
    data_name="temperature_BHE", data_unit="K", output_unit="°C", symbol="T"
)

all_variables = [v for v in locals().values() if isinstance(v, Variable)]


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


def normalize_vars(
    var1: str | Variable | None, var2: str | Variable | None, mesh: pv.DataSet
) -> tuple[Variable, Variable]:
    "Normalize arguments to return two Variables."
    default = ("time", "time") if "time" in [var1, var2] else "xyz"
    axes_idx = np.argwhere(
        np.invert(np.all(np.isclose(mesh.points, mesh.points[0]), axis=0))
    ).ravel()
    if len(axes_idx) == 0:
        axes_idx = [0, 1]
    match var1, var2:
        case None, None:
            if len(axes_idx) <= 1:
                axes_idx = [0, axes_idx[0] if axes_idx[0] != 0 else 1]
            x_var = Variable.find(default[axes_idx[0]], mesh)
            y_var = Variable.find(default[axes_idx[1]], mesh)
        case var1, None:
            x_var = Variable.find(default[axes_idx[0]], mesh)
            y_var = Variable.find(var1, mesh).magnitude  # type: ignore[arg-type]
        case None, var2:
            x_var = Variable.find(default[axes_idx[0]], mesh)
            y_var = Variable.find(var2, mesh).magnitude  # type: ignore[arg-type]
        case var1, var2:
            x_var = Variable.find(var1, mesh).magnitude  # type: ignore[arg-type]
            y_var = Variable.find(var2, mesh).magnitude  # type: ignore[arg-type]
    return x_var, y_var
