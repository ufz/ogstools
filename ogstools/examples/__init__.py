# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

import pyvista as pv

from ogstools.definitions import ROOT_DIR
from ogstools.meshlib import MeshSeries

from .analytical_solutions.steady_state_diffusion import analytical_diffusion

__all__ = ["analytical_diffusion"]

_examples_dir = ROOT_DIR / "examples"
_meshseries_dir = _examples_dir / "meshseries"
_feflow_dir = _examples_dir / "feflow"
_logs_dir = _examples_dir / "logs"
_prj_dir = _examples_dir / "prj"


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


fem_2layers_model = str(_feflow_dir / "2layers_model.fem")
fem_box_Neumann = str(_feflow_dir / "box_3D_neumann.fem")
fem_box_Robin = str(_feflow_dir / "box_3D_cauchy_areal.fem")
fem_2D_HT_model = str(_feflow_dir / "HT_toymodel_Diri.fem")

const_viscosity_thermal_convection_log = (
    _logs_dir / "ConstViscosityThermalConvection.log"
)
staggered_log = _logs_dir / "staggered_heat_transport_in_stationary_flow.log"
parallel_log = _logs_dir / "steady_state_diffusion_parallel.log"
parallel_1_info = _logs_dir / "parallel_1_info.txt"
parallel_3_debug = _logs_dir / "parallel_3_debug.txt"
serial_convergence_long = _logs_dir / "serial_convergence_long.txt"
serial_convergence_short = _logs_dir / "serial_convergence_short.txt"
serial_critical = _logs_dir / "serial_critical.txt"
serial_info = _logs_dir / "serial_info.txt"
serial_time_step_rejected = _logs_dir / "serial_time_step_rejected.txt"
serial_warning_only = _logs_dir / "serial_warning_only.txt"

steady_state_diffusion_prj = _prj_dir / "steady_state_diffusion.prj"
nuclear_decay_prj = _prj_dir / "nuclear_decay.prj"
nuclear_decay_bc = _prj_dir / "decay_boundary_conditions.py"
