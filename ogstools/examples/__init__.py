# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

import pyvista as pv

from ogstools.definitions import EXAMPLES_DIR
from ogstools.meshlib import MeshSeries

from .analytical_solutions.steady_state_diffusion import analytical_diffusion

__all__ = ["analytical_diffusion"]

_meshseries_dir = EXAMPLES_DIR / "meshseries"
_feflow_dir = EXAMPLES_DIR / "feflow"
_logs_dir = EXAMPLES_DIR / "logs"
_prj_dir = EXAMPLES_DIR / "prj"


def load_meshseries_THM_2D_PVD():
    return MeshSeries(str(_meshseries_dir / "2D.pvd"), time_unit="s")


def load_meshseries_CT_2D_XDMF():
    return MeshSeries(str(_meshseries_dir / "elder.xdmf"), time_unit="s")


def load_meshseries_HT_2D_XDMF():
    return MeshSeries(
        str(_meshseries_dir / "2D_single_fracture_HT_2D_single_fracture.xdmf"),
        time_unit="s",
    )


def load_mesh_mechanics_2D():
    return pv.XMLUnstructuredGridReader(
        str(_meshseries_dir / "mechanics_example.vtu")
    ).read()


feflow_model_2layers = _feflow_dir / "2layers_model.fem"
feflow_model_box_Neumann = _feflow_dir / "box_3D_neumann.fem"
feflow_model_box_Robin = _feflow_dir / "box_3D_cauchy_areal.fem"
feflow_model_box_well_BC = _feflow_dir / "box_3D_wellBC.fem"
feflow_model_2D_HT_model = _feflow_dir / "HT_toymodel_Diri.fem"
feflow_model_2D_CT_t_560 = _feflow_dir / "CT_2D_line_560.fem"
feflow_model_2D_CT_t_168 = _feflow_dir / "CT_2D_line_168.fem"
feflow_model_2D_CT_t_28 = _feflow_dir / "CT_2D_line_28.fem"

log_const_viscosity_thermal_convection = (
    _logs_dir / "ConstViscosityThermalConvection.log"
)
log_staggered = _logs_dir / "staggered_heat_transport_in_stationary_flow.log"
log_parallel = _logs_dir / "steady_state_diffusion_parallel.log"
info_parallel_1 = _logs_dir / "parallel_1_info.txt"
debug_parallel_3 = _logs_dir / "parallel_3_debug.txt"
serial_convergence_long = _logs_dir / "serial_convergence_long.txt"
serial_convergence_short = _logs_dir / "serial_convergence_short.txt"
serial_critical = _logs_dir / "serial_critical.txt"
serial_info = _logs_dir / "serial_info.txt"
serial_time_step_rejected = _logs_dir / "serial_time_step_rejected.txt"
serial_warning_only = _logs_dir / "serial_warning_only.txt"

prj_steady_state_diffusion = _prj_dir / "steady_state_diffusion.prj"
prj_nuclear_decay = _prj_dir / "nuclear_decay.prj"
pybc_nuclear_decay = _prj_dir / "decay_boundary_conditions.py"
