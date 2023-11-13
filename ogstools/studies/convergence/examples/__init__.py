from importlib import resources

from .steady_state_diffusion import (
    analytical_solution as steady_state_diffusion_analytical_solution,
)

_prefix = resources.files(__name__)
steady_state_diffusion_prj = _prefix / "steady_state_diffusion.prj"
nuclear_decay_prj = _prefix / "nuclear_decay.prj"
nuclear_decay_bc = _prefix / "decay_boundary_conditions.py"

__all__ = [
    "steady_state_diffusion_analytical_solution",
    "steady_state_diffusion_prj",
    "nuclear_decay_prj",
    "nuclear_decay_bc",
]
