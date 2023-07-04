# from types import SimpleNamespace

from dataclasses import dataclass

from pint import UnitRegistry

ureg: UnitRegistry = UnitRegistry()
Q_ = ureg.Quantity


@dataclass
class UnitSetup:
    time: str
    power: str


units = UnitSetup(time="s", power="W")
