"""Defines derived PropertyCollection classes for uncoupled processes.

Each class inherits from the PropertyCollection class and provides
Property attributes for the commonly found data in the corresponding
processes.
"""

from dataclasses import dataclass

from . import _engfuncs as ef
from .property import MatrixProperty, ScalarProperty, VectorProperty
from .property_collection import PropertyCollection


@dataclass(init=False)
class T(PropertyCollection):
    """Property Collection for the T process."""

    temperature: ScalarProperty
    heatflowrate: ScalarProperty

    def __init__(self):
        """Initialize the PropertyCollection with default attributes."""
        super().__init__()
        self.temperature = ScalarProperty(
            "temperature", "K", "°C", "temperature", "temperature_active"
        )
        self.heatflowrate = ScalarProperty(
            "HeatFlowRate", "", "", "HeatFlowRate", "temperature_active"
        )


@dataclass(init=False)
class H(PropertyCollection):
    """Property Collection for the H process."""

    pressure: ScalarProperty
    velocity: VectorProperty
    massflowrate: ScalarProperty

    def __init__(self):
        """Initialize the PropertyCollection with default attributes."""
        super().__init__()
        self.pressure = ScalarProperty(
            "pressure", "Pa", "MPa", "pore pressure", "pressure_active"
        )
        self.velocity = VectorProperty(
            "velocity", "m/s", "mm/d", "darcy velocity", "pressure_active"
        )
        self.massflowrate = ScalarProperty(
            "MassFlowRate", "", "", "MassFlowRate", "pressure_active"
        )


@dataclass(init=False)
class M(PropertyCollection):
    """Property Collection for the M process."""

    displacement: VectorProperty
    strain: MatrixProperty
    stress: MatrixProperty
    von_mises_stress: ScalarProperty
    effective_pressure: ScalarProperty
    qp_ratio: ScalarProperty
    nodal_forces: VectorProperty

    def __init__(self):
        """Initialize the PropertyCollection with default attributes."""
        super().__init__()
        self.displacement = VectorProperty(
            "displacement", "m", "mm", "displacement", "displacement_active"
        )
        self.strain = MatrixProperty(
            "epsilon", "", "percent", "strain", "displacement_active"
        )
        self.stress = MatrixProperty(
            "sigma", "Pa", "MPa", "stress", "displacement_active"
        )
        self.von_mises_stress = self.stress(
            output_name="von Mises stress", func=ef.von_mises
        )
        self.effective_pressure = self.stress(
            output_name="effective pressure", func=ef.effective_pressure
        )
        self.qp_ratio = self.stress(
            output_name="QP ratio", output_unit="percent", func=ef.qp_ratio
        )
        self.nodal_forces = VectorProperty(
            "NodalForces", "", "", "NodalForces", "displacement_active"
        )
