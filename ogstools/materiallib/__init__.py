# ogstools/materiallib/__init__.py

# from .core.core import MaterialList, MaterialDB, MaterialLib, Medium, Property
# from .validation import validate_medium

# ogstools/materiallib/__init__.py

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
