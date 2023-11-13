# flake8: noqa: E501
"Predefined properties."

from typing import Optional

from . import vector2scalar as v2s
from .property import Matrix, Property, Scalar, Vector

T_mask = "temperature_active"
H_mask = "pressure_active"
M_mask = "displacement_active"

# fmt: off
material_id = Scalar("MaterialIDs", categoric=True)
displacement = Vector("displacement", "m", "m", mask=M_mask)
effective_pressure = Scalar("sigma", "Pa", "MPa", "effective_pressure", M_mask, v2s.effective_pressure)
heatflowrate = Scalar("HeatFlowRate", mask=T_mask)
massflowrate = Scalar("MassFlowRate", mask=H_mask)
nodal_forces = Vector("NodalForces", mask=M_mask)
pressure = Scalar("pressure", "Pa", "MPa", "pore_pressure", H_mask)
hydraulic_height = Scalar("pressure", "m", "m", "hydraulic_height", H_mask)
qp_ratio = Scalar("sigma", "Pa", "percent", "QP_ratio", M_mask, v2s.qp_ratio)
strain = Matrix("epsilon", "", "percent", "strain", M_mask)
stress = Matrix("sigma", "Pa", "MPa", "stress", M_mask)
temperature = Scalar("temperature", "K", "°C", mask=T_mask)
velocity = Vector("velocity", "m/s", "m/s", "darcy_velocity", H_mask)
von_mises_stress = Scalar("sigma", "Pa", "MPa", "von_Mises_stress", M_mask, v2s.von_mises)
# fmt: on

all_properties = [v for v in locals().values() if isinstance(v, Property)]


def find_property_preset(property_name: str) -> Optional[Property]:
    """Return predefined property with given output_name."""
    for prop in all_properties:
        if prop.output_name == property_name:
            return prop
    # if not found by output name, find by data_name
    for prop in all_properties:
        if prop.data_name == property_name:
            return prop
    return None


def _resolve_property(property_name: str, shape: tuple) -> Property:
    if found_property := find_property_preset(property_name):
        return found_property
    if len(shape) == 1:
        return Scalar(property_name)
    if shape[1] in [2, 3]:
        return Vector(property_name)
    return Matrix(property_name)
