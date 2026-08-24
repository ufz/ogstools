# SPDX-FileCopyrightText: Copyright (c) OpenGeoSys Community (opengeosys.org)
# SPDX-License-Identifier: BSD-3-Clause

"""Predefined variables for data and unit transformation."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import pyvista as pv

from . import integrity, tensor_math
from .custom_colormaps import integrity_cmap, none_cmap, temperature_cmap
from .func import Function
from .matrix import Matrix
from .unit_registry import u_reg
from .variable import Scalar, Variable
from .vector import Components_BHE, Vector

__all__ = ["Matrix", "Scalar", "Variable", "Vector", "u_reg"]

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
    output_unit="%",
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
dilatancy_critescu = Scalar(
    data_name="sigma",
    data_unit="",
    output_unit="",
    output_name="dilatancy_criterion",
    symbol=r"F_\mathrm{dil}",
    func=Function(
        integrity.dilatancy_critescu,
        ["pressure"],
        {"a": -0.01697, "b": 0.8996},
    ),
    mask=M_MASK,
    color=COLOR_MECH,
    cmap=integrity_cmap,
    bilinear_cmap=True,
)
dilatancy_critescu_eff = dilatancy_critescu.replace(
    output_name="effective_dilatancy_criterion",
    func=Function(
        integrity.dilatancy_critescu, [], {"a": -0.01697, "b": 0.8996}
    ),
)

dilatancy_alkan = Scalar(
    data_name="sigma",
    data_unit="Pa",
    output_unit="MPa",
    output_name="dilatancy_criterion",
    symbol=r"F_\mathrm{dil}",
    func=Function(
        integrity.dilatancy_alkan, ["pressure"], {"b": 0.04, "tau_max": 33e6}
    ),
    mask=M_MASK,
    color=COLOR_MECH,
    cmap=integrity_cmap,
    bilinear_cmap=True,
)
dilatancy_alkan_eff = dilatancy_alkan.replace(
    output_name="effective_dilatancy_criterion",
    func=Function(integrity.dilatancy_alkan, [], {"b": 0.04, "tau_max": 33e6}),
)

fluid_pressure_criterion = Scalar(
    data_name="sigma",
    data_unit="Pa",
    output_unit="MPa",
    output_name="fluid_pressure_criterion",
    symbol=r"\sigma_{III}'",
    func=Function(
        integrity.fluid_pressure_criterion, ["pressure"], {"biot": 1.0}
    ),
    mask=M_MASK,
    color=COLOR_MECH,
    cmap=integrity_cmap,
    bilinear_cmap=True,
)
nodal_forces = Vector(data_name="NodalForces", mask=M_MASK)

# ====== other ======
saturation = Scalar(
    data_name="Si",
    data_unit="",
    output_unit="%",
    output_name="saturation",
    symbol="s",
)

temperature_BHE = Components_BHE(
    data_name="temperature_BHE",
    data_unit="K",
    output_unit="°C",
    symbol="T",
)

time = Scalar("timevalues", "s", "s", output_name="time", symbol="t")

points = Vector("points", "m", "m", output_name="", color="k")

none = Scalar("None", output_name="", cmap=none_cmap, categoric=True, mask="")

all_variables = [v for v in locals().values() if isinstance(v, Variable)]


def get_dataframe() -> pd.DataFrame:
    data = [
        [
            "preset",
            "data_name",
            "data_unit",
            "output_unit",
            "output_name",
            "type",
        ]
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


def _normalize_vars(
    var1: str | Variable | None,
    var2: str | Variable | None,
    dataset: pv.DataSet | Sequence[pv.DataSet],
    default: str | list[str],
) -> tuple[Variable, Variable]:
    "Normalize arguments to return two Variables."
    mesh = dataset[0] if isinstance(dataset, Sequence) else dataset
    axes_idx = np.argwhere(
        np.invert(np.all(np.isclose(mesh.points, mesh.points[0]), axis=0))
    ).ravel()
    if len(axes_idx) == 0:
        axes_idx = [0, 1]
    match var1, var2:
        case None, None:
            if len(axes_idx) <= 1:
                axes_idx = [0, axes_idx[0] if axes_idx[0] != 0 else 1]
            x_var = Variable.find(default[axes_idx[0]], dataset)
            y_var = Variable.find(default[axes_idx[1]], dataset)
        case var1, None:
            x_var = Variable.find(default[axes_idx[0]], dataset)
            y_var = Variable.find(var1, dataset).magnitude  # type: ignore[arg-type]
        case None, var2:
            x_var = Variable.find(default[axes_idx[0]], dataset)
            y_var = Variable.find(var2, dataset).magnitude  # type: ignore[arg-type]
        case var1, var2:
            x_var = Variable.find(var1, dataset).magnitude  # type: ignore[arg-type]
            y_var = Variable.find(var2, dataset).magnitude  # type: ignore[arg-type]
    return x_var, y_var
