# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

# flake8: noqa: E501
"""Predefined properties.

".. seealso:: :py:mod:`ogstools.propertylib.tensor_math`"
"""

from functools import partial
from typing import Union

import pandas as pd
import pyvista as pv

from . import mesh_dependent, tensor_math
from .custom_colormaps import integrity_cmap, temperature_cmap
from .matrix import Matrix
from .property import Property, Scalar
from .tensor_math import identity
from .vector import Vector

T_MASK = "temperature_active"
H_MASK = "pressure_active"
M_MASK = "displacement_active"

# Default style for meshplotlib plots
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
    color=group_color_thermal,
    linestyle=line_styles[0],
)
heatflowrate = Scalar(
    data_name="HeatFlowRate",
    mask=T_MASK,
    color=group_color_thermal,
    linestyle=line_styles[1],
)

# hydraulic
pressure = Scalar(
    data_name="pressure",
    data_unit="Pa",
    output_unit="MPa",
    output_name="pore_pressure",
    mask=H_MASK,
    cmap="Blues",
    color=group_color_hydraulic,
    linestyle=line_styles[0],
)
hydraulic_height = Scalar(
    data_name="pressure",
    data_unit="m",
    output_unit="m",
    output_name="hydraulic_height",
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
    color=group_color_mechanical,
    linestyle=line_styles[0],
)
dilatancy_critescu_tot = Scalar(
    data_name="sigma",
    data_unit="Pa",
    output_unit="",
    output_name="dilatancy_criterion",
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
    mask=M_MASK,
    func=mesh_dependent.fluid_pressure_criterion,
    mesh_dependent=True,
    cmap=integrity_cmap,
    bilinear_cmap=True,
    color=group_color_mechanical,
    linestyle=line_styles[5],
)
nodal_forces = Vector(data_name="NodalForces", mask=M_MASK)


all_properties = [v for v in locals().values() if isinstance(v, Property)]


def get_preset(
    mesh_property: Union[Property, str], mesh: pv.UnstructuredGrid
) -> Property:
    """
    Returns a Property preset or creates one with correct type.

    Searches for presets by data_name and output_name and returns if found.
    If 'mesh_property' is given as type Property this will also look for
    derived properties (difference, aggregate).
    Otherwise create Scalar, Vector, or Matrix Property depending on the shape
    of data in mesh.

    :param mesh_property:   The property to retrieve or its name if a string.
    :param mesh:            The mesh containing the property data.
    :returns: A corresponding Property preset or a new Property of correct type.
    """
    data_keys: list[str] = list(set().union(mesh.point_data, mesh.cell_data))
    error_msg = (
        f"Data not found in mesh. Available data names are {data_keys}. "
    )

    if isinstance(mesh_property, Property):
        if mesh_property.data_name in data_keys:
            return mesh_property
        matches = [
            mesh_property.output_name in data_key for data_key in data_keys
        ]
        if not any(matches):
            raise KeyError(error_msg)
        data_key = data_keys[matches.index(True)]
        if data_key == f"{mesh_property.output_name}_difference":
            return mesh_property.difference
        return mesh_property.replace(
            data_name=data_key,
            data_unit=mesh_property.output_unit,
            output_unit=mesh_property.output_unit,
            output_name=data_key,
            func=identity,
            mesh_dependent=False,
        )

    for prop in all_properties:
        if prop.output_name == mesh_property:
            return prop
    for prop in all_properties:
        if prop.data_name == mesh_property:
            return prop

    matches = [mesh_property in data_key for data_key in data_keys]
    if not any(matches):
        raise KeyError(error_msg)

    data_shape = mesh[mesh_property].shape
    if len(data_shape) == 1:
        return Scalar(mesh_property)
    if data_shape[1] in [2, 3]:
        return Vector(mesh_property)
    return Matrix(mesh_property)


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
