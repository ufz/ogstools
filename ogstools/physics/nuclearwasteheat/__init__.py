# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

# Author: Florian Zill (Helmholtz Centre for Environmental Research GmbH - UFZ)
# Co-Author: Christian B. Silbermann (TU Bergakademie Freiberg)
"Provide proxy models to calculate heat generated by nuclear waste."

from .defaults import (
    repo_2020,
    repo_2020_conservative,
    repo_be_ha_2016,
    waste_types,
)

__all__ = [
    "repo_2020",
    "repo_2020_conservative",
    "repo_be_ha_2016",
    "waste_types",
]
