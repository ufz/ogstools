# Author: Florian Zill (Helmholtz Centre for Environmental Research GmbH - UFZ)
"""Define easy-to-access Property classes and PropertyCollection instances."""

from . import _coupled, _uncoupled, defaults
from .property import MatrixProperty, Property, ScalarProperty, VectorProperty
from .property_collection import PropertyCollection

T = _uncoupled.T()
H = _uncoupled.H()
M = _uncoupled.M()

TH = _coupled.TH()
HM = _coupled.HM()
TM = _coupled.TM()
THM = _coupled.THM()

processes: list[PropertyCollection] = [T, H, M, TH, HM, TM, THM]


__all__ = [
    "defaults",
    "Property",
    "ScalarProperty",
    "VectorProperty",
    "MatrixProperty",
]
