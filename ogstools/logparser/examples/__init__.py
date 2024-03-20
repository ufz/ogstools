from importlib import resources

_prefix = resources.files(__name__)
liquid_flow_log = _prefix / "ogs.log"
const_viscosity_thermal_convection_log = (
    _prefix / "ConstViscosityThermalConvection.log"
)
staggered_log = "staggered_heat_transport_in_stationary_flow.log"
