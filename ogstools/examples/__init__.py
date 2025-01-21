# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

from ogstools.definitions import EXAMPLES_DIR
from ogstools.meshlib import Mesh, MeshSeries

from .analytical_solutions.steady_state_diffusion import analytical_diffusion

__all__ = ["analytical_diffusion"]

_meshseries_dir = EXAMPLES_DIR / "meshseries"
_feflow_dir = EXAMPLES_DIR / "feflow"
_logs_dir = EXAMPLES_DIR / "logs"
_msh_dir = EXAMPLES_DIR / "msh"
_prj_dir = EXAMPLES_DIR / "prj"
_surface_dir = EXAMPLES_DIR / "meshlib" / "mesh1" / "surface_data"
_shp_dir = EXAMPLES_DIR / "shapefiles"


def load_meshseries_THM_2D_PVD():
    return MeshSeries(str(_meshseries_dir / "2D.pvd"))


def load_meshseries_CT_2D_XDMF():
    return MeshSeries(str(_meshseries_dir / "elder.xdmf"))


def load_meshseries_HT_2D_XDMF():
    return MeshSeries(
        str(_meshseries_dir / "2D_single_fracture_HT_2D_single_fracture.xdmf")
    )


def load_meshseries_HT_2D_PVD():
    return MeshSeries(
        str(_meshseries_dir / "2D_single_fracture_HT_2D_single_fracture.pvd")
    )


def load_meshseries_HT_2D_VTU():
    return MeshSeries(
        str(
            _meshseries_dir
            / "2D_single_fracture_HT_2D_single_fracture"
            / "2D_single_fracture_HT_2D_single_fracture_0_96.vtu"
        )
    )


def load_meshseries_HT_2D_paraview_XMF():
    return MeshSeries(
        str(_meshseries_dir / "2D_single_fracture_HT_2D_single_fracture.xmf")
    )


def load_mesh_mechanics_2D():
    return Mesh.read(_meshseries_dir / "mechanics_example.vtu")


msh_geolayers_2d = _msh_dir / "geolayers_2d.msh"
msh_geoterrain_3d = _msh_dir / "geoterrain_3d.msh"

feflow_model_box_IOFLOW = _feflow_dir / "box_3D_cauchy_areal_IO_FLOW.fem"
feflow_model_box_Neumann = _feflow_dir / "box_3D_neumann.fem"
feflow_model_box_Robin = _feflow_dir / "box_3D_cauchy_areal.fem"
feflow_model_box_well_BC = _feflow_dir / "box_3D_wellBC.fem"
feflow_model_2D_HT = _feflow_dir / "HT_toymodel_Diri.fem"
feflow_model_2D_HT_hetero = _feflow_dir / "HT_toymodel_Diri_hetero.fem"
feflow_model_2D_CT_t_560 = _feflow_dir / "CT_2D_line_560.fem"
feflow_model_2D_CT_t_168 = _feflow_dir / "CT_2D_line_168.fem"
feflow_model_2D_CT_t_28 = _feflow_dir / "CT_2D_line_28.fem"
feflow_model_2D_HTC = _feflow_dir / "HTC.fem"
feflow_model_2D_HTA = _feflow_dir / "HTA.fem"

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

##

prj_aniso_expansion = _prj_dir / "aniso_expansion.prj"
prj_deactivate_replace = _prj_dir / "deactivate_replace.prj"
prj_mechanics = _prj_dir / "mechanics.prj"
prj_steady_state_diffusion = _prj_dir / "steady_state_diffusion.prj"
prj_nuclear_decay = _prj_dir / "nuclear_decay.prj"
prj_tunnel_trm = _prj_dir / "tunnel_trm.prj"
prj_tunnel_trm_withincludes = _prj_dir / "tunnel_trm_withincludes.prj"
prj_trm_from_scratch = _prj_dir / "trm_from_scratch.prj"
prj_include_solid = _prj_dir / "solid_inc.xml"
prj_include_solid_ref = _prj_dir / "solid.xml"
prj_heat_transport = _prj_dir / "HeatTransportBHE_ogs6py.prj"
prj_heat_transport_bhe_simple = _prj_dir / "HeatTransportBHE_simple.prj"
prj_staggered = _prj_dir / "hm1_3Dcube.prj"
prj_staggered_ref = _prj_dir / "hm1_3Dcube_ref.prj"
prj_pid_timestepping = _prj_dir / "PID_timestepping.prj"
prj_pid_timestepping_ref = _prj_dir / "PID_timestepping_ref.prj"
prj_solid_inc_ref = _prj_dir / "tunnel_ogs6py_solid_inc_ref.prj"
prj_time_dep_het_param = (
    _prj_dir / "TimeDependentHeterogeneousBoundaryConditions.prj"
)
prj_time_dep_het_param_ref = (
    _prj_dir / "TimeDependentHeterogeneousBoundaryConditions_ref.prj"
)
prj_beier_sandbox = _prj_dir / "beier_sandbox.prj"
prj_beier_sandbox_ref = _prj_dir / "beier_sandbox_ref.prj"
prj_square_1e4_robin = _prj_dir / "square_1e4_robin.prj"
prj_square_1e4_robin_ref = _prj_dir / "square_1e4_robin_ref.prj"
pybc_nuclear_decay = _prj_dir / "decay_boundary_conditions.py"


surface_paths = [
    _surface_dir / (file + ".vtu")
    for file in ["00_KB", "01_q", "02_krl", "03_S3", "04_krp"]
]

test_shapefile = _shp_dir / "test_shape.shp"
circle_shapefile = _shp_dir / "circle.shp"

elder_h5 = _meshseries_dir / "elder.h5"
elder_xdmf = _meshseries_dir / "elder.xdmf"
mechanics_vtu = _meshseries_dir / "mechanics_example.vtu"
