# Author: Florian Zill (Helmholtz Centre for Environmental Research GmbH - UFZ)
"""Define easy-to-access Property classes and PropertyCollection instances."""

from . import _coupled, _uncoupled, property_collection
from .property import MatrixProperty, ScalarProperty, VectorProperty

material_id = property_collection.PropertyCollection().material_id

T = _uncoupled.T()
H = _uncoupled.H()
M = _uncoupled.M()

TH = _coupled.TH()
HM = _coupled.HM()
TM = _coupled.TM()
THM = _coupled.THM()


__all__ = ["ScalarProperty", "VectorProperty", "MatrixProperty"]
