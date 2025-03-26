# ogstools/materiallib/__init__.py

# from .core.core import MaterialList, MaterialDB, MaterialLib, Medium, Property
# from .validation import validate_medium

# ogstools/materiallib/__init__.py

from .core import (
    #MaterialLib,
    MaterialList,
    MaterialDB,
    Medium,
    Phase,
    Component,
    Property
)

from .validation import validate_medium

__all__ = [
    #"MaterialLib",
    "MaterialList",
    "MaterialDB",
    "Medium",
    "Phase",
    "Component",
    "Property",
    "validate_medium"
]
