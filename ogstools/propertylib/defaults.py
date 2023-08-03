# flake8: noqa: E501
"Predefined properties."

from . import vector2scalar as v2s
from .property import MatrixProperty as Matrix
from .property import Property, TagType
from .property import ScalarProperty as Scalar
from .property import VectorProperty as Vector

T_mask = "temperature_active"
H_mask = "pressure_active"
M_mask = "displacement_active"

# fmt: off
displacement = Vector("displacement", "m", "m", mask=M_mask)
effective_pressure = Scalar("sigma", "Pa", "MPa", "effective_pressure", M_mask, v2s.effective_pressure, TagType.unit_dim_const)
heatflowrate = Scalar("HeatFlowRate", mask=T_mask)
massflowrate = Scalar("MassFlowRate", mask=H_mask)
nodal_forces = Vector("NodalForces", mask=M_mask)
pressure = Scalar("pressure", "Pa", "MPa", "pore_pressure", H_mask)
qp_ratio = Scalar("sigma", "Pa", "percent", "QP_ratio", M_mask, v2s.qp_ratio)
strain = Matrix("epsilon", "", "percent", "strain", M_mask)
stress = Matrix("sigma", "Pa", "MPa", "stress", M_mask)
temperature = Scalar("temperature", "K", "°C", mask=T_mask)
velocity = Vector("velocity", "m/s", "m/s", "darcy_velocity", H_mask)
von_mises_stress = Scalar("sigma", "Pa", "MPa", "von_Mises_stress", M_mask, v2s.von_mises, TagType.unit_dim_const)
# fmt: on

all_properties = [v for v in locals().values() if isinstance(v, Property)]
