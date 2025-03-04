# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

from ._heat_conduction import heat_conduction_temperature
from ._steady_state_diffusion import diffusion_head_analytical

__all__ = [
    "diffusion_head_analytical",
    "heat_conduction_temperature",
]
