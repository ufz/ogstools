"""Defines derived PropertyCollection classes for coupled processes.

Each class inherits the attributes from the classes corresponding to the
uncoupled processes.
"""

from dataclasses import dataclass

from ._uncoupled import H, M, T


@dataclass(init=False)
class TH(T, H):
    """Property Collection for the TH process."""

    def __init__(self):
        """Initialize the PropertyCollection with default attributes."""
        super().__init__()


@dataclass(init=False)
class HM(H, M):
    """Property Collection for the HM process."""

    def __init__(self):
        """Initialize the PropertyCollection with default attributes."""
        super().__init__()


@dataclass(init=False)
class TM(T, M):
    """Property Collection for the TM process."""

    def __init__(self):
        """Initialize the PropertyCollection with default attributes."""
        super().__init__()


@dataclass(init=False)
class THM(T, H, M):
    """Property Collection for the THM process."""

    def __init__(self):
        """Initialize the PropertyCollection with default attributes."""
        super().__init__()
