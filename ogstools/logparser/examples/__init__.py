# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

from importlib import resources

_prefix = resources.files(__name__)
const_viscosity_thermal_convection_log = (
    _prefix / "ConstViscosityThermalConvection.log"
)
staggered_log = _prefix / "staggered_heat_transport_in_stationary_flow.log"
parallel_log = _prefix / "steady_state_diffusion_parallel.log"

parallel_1_info = _prefix / "parallel_1_info.txt"
parallel_3_debug = _prefix / "parallel_3_debug.txt"
serial_convergence_long = _prefix / "serial_convergence_long.txt"
serial_convergence_short = _prefix / "serial_convergence_short.txt"
serial_critical = _prefix / "serial_critical.txt"
serial_info = _prefix / "serial_info.txt"
serial_time_step_rejected = _prefix / "serial_time_step_rejected.txt"
serial_warning_only = _prefix / "serial_warning_only.txt"
