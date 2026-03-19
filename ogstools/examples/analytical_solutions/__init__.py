# SPDX-FileCopyrightText: Copyright (c) OpenGeoSys Community (opengeosys.org)
# SPDX-License-Identifier: BSD-3-Clause

from ._heat_conduction import heat_conduction_temperature
from ._steady_state_diffusion import diffusion_head_analytical

__all__ = [
    "diffusion_head_analytical",
    "heat_conduction_temperature",
]
