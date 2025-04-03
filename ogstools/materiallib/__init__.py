# ogstools/materiallib/__init__.py

# from .core.core import MaterialList, MaterialDB, MaterialLib, Medium, Property
# from .validation import validate_medium

# ogstools/materiallib/__init__.py

import logging

from .core import (
    Component,
    Components,
    MaterialDB,
    # MaterialLib,
    MaterialList,
    Medium,
    Phase,
    Property,
)
from .validation import validate_medium

logging.basicConfig(
    level=logging.INFO,  # could be DEBUG or WARNING later
    format="%(levelname)s: %(message)s",
)


def set_log_level(level: str = "INFO") -> None:
    """Change the global log level for materiallib."""
    logging.getLogger().setLevel(level.upper())


__all__ = [
    # "MaterialLib",
    "MaterialList",
    "MaterialDB",
    "Medium",
    "Phase",
    "Component",
    "Components",
    "Property",
    "validate_medium",
]
