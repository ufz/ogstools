# flake8: noqa: E501
"""Predefined properties.

".. seealso:: :py:mod:`ogstools.propertylib.tensor_math`"
"""

from functools import partial
from typing import Optional

import pandas as pd

from . import mesh_dependent, tensor_math
from .custom_colormaps import integrity_cmap, temperature_cmap
from .matrix import Matrix
from .property import Property, Scalar
from .vector import Vector

T_MASK = "temperature_active"
H_MASK = "pressure_active"
M_MASK = "displacement_active"

# general
material_id = Scalar(data_name="MaterialIDs", categoric=True, cmap="tab20")

# thermal
temperature = Scalar(
    data_name="temperature",
    data_unit="K",
    output_name="temperature",
    output_unit="°C",
    mask=T_MASK,
    cmap=temperature_cmap,
    bilinear_cmap=True,
)
heatflowrate = Scalar(data_name="HeatFlowRate", mask=T_MASK)

# hydraulic
pressure = Scalar(
    data_name="pressure",
    data_unit="Pa",
    output_unit="MPa",
    output_name="pore_pressure",
    mask=H_MASK,
    cmap="Blues",
)
hydraulic_height = Scalar(
    data_name="pressure",
    data_unit="m",
    output_unit="m",
    output_name="hydraulic_height",
    mask=H_MASK,
    cmap="Blues",
)
velocity = Vector(
    data_name="velocity",
    data_unit="m/s",
    output_unit="m/s",
    output_name="darcy_velocity",
    mask=H_MASK,
)
massflowrate = Scalar(data_name="MassFlowRate", mask=H_MASK)

# mechanical
displacement = Vector(
    data_name="displacement",
    data_unit="m",
    output_unit="m",
    mask=M_MASK,
    cmap="PRGn",
    bilinear_cmap=True,
)
strain = Matrix(
    data_name="epsilon",
    data_unit="",
    output_unit="percent",
    output_name="strain",
    mask=M_MASK,
)
stress = Matrix(
    data_name="sigma",
    data_unit="Pa",
    output_unit="MPa",
    output_name="stress",
    mask=M_MASK,
)
effective_pressure = Scalar(
    data_name="sigma",
    data_unit="Pa",
    output_unit="MPa",
    output_name="effective_pressure",
    mask=M_MASK,
    func=tensor_math.effective_pressure,
)
dilatancy_critescu = Scalar(
    data_name="sigma",
    data_unit="Pa",
    output_unit="",
    output_name="dilatancy_criterion",
    mask=M_MASK,
    func=mesh_dependent.dilatancy_critescu,
    mesh_dependent=True,
    cmap=integrity_cmap,
    bilinear_cmap=True,
)
dilatancy_critescu_eff = dilatancy_critescu.replace(
    output_name="effective_dilatancy_criterion",
    func=partial(mesh_dependent.dilatancy_critescu, effective=True),
)

dilatancy_alkan = Scalar(
    data_name="sigma",
    data_unit="Pa",
    output_unit="MPa",
    output_name="dilatancy_criterion",
    mask=M_MASK,
    func=mesh_dependent.dilatancy_alkan,
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
    mask=M_MASK,
    func=mesh_dependent.fluid_pressure_criterion,
    mesh_dependent=True,
    cmap=integrity_cmap,
    bilinear_cmap=True,
)
nodal_forces = Vector(data_name="NodalForces", mask=M_MASK)


all_properties = [v for v in locals().values() if isinstance(v, Property)]


def get_preset(property_name: str, shape: Optional[tuple] = None) -> Property:
    """
    Returns a Property preset or create one with correct type.

    Searches for presets by data_name and output_name and returns if found.
    Otherwise create Scalar, Vector or Matrix Property depending on shape.
    """
    for prop in all_properties:
        if prop.output_name == property_name:
            return prop
    for prop in all_properties:
        if prop.data_name == property_name:
            return prop
    if shape is None or len(shape) == 1:
        return Scalar(property_name)
    if shape[1] in [2, 3]:
        return Vector(property_name)
    return Matrix(property_name)


def get_dataframe() -> pd.DataFrame:
    data = [
        "preset,data_name,data_unit,output_unit,output_name,type".split(",")
    ]
    for preset_name, preset_value in globals().items():
        if isinstance(preset := preset_value, Property):
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
