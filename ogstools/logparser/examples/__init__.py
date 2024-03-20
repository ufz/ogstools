from importlib import resources

_prefix = resources.files(__name__)
const_viscosity_thermal_convection_log = (
    _prefix / "ConstViscosityThermalConvection.log"
)
staggered_log = _prefix / "staggered_heat_transport_in_stationary_flow.log"
parallel_log = _prefix / "steady_state_diffusion_parallel.log"
