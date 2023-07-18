"""Defines derived PropertyCollection classes for uncoupled processes.

Each class inherits from the PropertyCollection class and provides
Property attributes for the commonly found data in the corresponding
processes.
"""

from dataclasses import dataclass

from . import defaults
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
        self.temperature = defaults.temperature
        self.heatflowrate = defaults.heatflowrate


@dataclass(init=False)
class H(PropertyCollection):
    """Property Collection for the H process."""

    pressure: ScalarProperty
    velocity: VectorProperty
    massflowrate: ScalarProperty

    def __init__(self):
        """Initialize the PropertyCollection with default attributes."""
        super().__init__()
        self.pressure = defaults.pressure
        self.velocity = defaults.velocity
        self.massflowrate = defaults.massflowrate


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
        self.displacement = defaults.displacement
        self.strain = defaults.strain
        self.stress = defaults.stress
        self.von_mises_stress = defaults.von_mises_stress
        self.effective_pressure = defaults.effective_pressure
        self.qp_ratio = defaults.qp_ratio
        self.nodal_forces = defaults.nodal_forces
