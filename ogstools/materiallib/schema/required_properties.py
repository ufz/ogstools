# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

from typing import Any


def required_property_names(schema: dict[str, Any]) -> set[str]:
    """
    Collect all required property names from a process schema.
    Includes medium-level, phase-level and component-level properties.
    """
    required: set[str] = set()
    PHASES_WITH_COMPONENTS = {"AqueousLiquid", "Gas", "NonAqueousLiquid"}

    # Medium-level
    medium_properties = schema.get("properties", [])
    required.update(medium_properties)

    # Phase + components
    for phase in schema.get("phases", []):
        required.update(phase.get("properties", []))
        if phase.get("type") in PHASES_WITH_COMPONENTS:
            for component_props in phase.get("components", {}).values():
                required.update(component_props)

    return required
