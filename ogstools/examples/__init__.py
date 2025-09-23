# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

from typing import Literal

import numpy as np
import pyvista as pv

from ogstools.definitions import EXAMPLES_DIR
from ogstools.meshlib import Mesh, MeshSeries

from . import analytical_solutions as anasol

_meshseries_dir = EXAMPLES_DIR / "meshseries"
_feflow_dir = EXAMPLES_DIR / "feflow"
_logs_dir = EXAMPLES_DIR / "logs"
_msh_dir = EXAMPLES_DIR / "msh"
_prj_dir = EXAMPLES_DIR / "prj"
_surface_dir = EXAMPLES_DIR / "meshlib" / "mesh1" / "surface_data"
_shp_dir = EXAMPLES_DIR / "shapefiles"
_yaml_mesh_dir = EXAMPLES_DIR / "meshlib" / "meshes_from_yaml"


def load_meshseries_THM_2D_PVD():
    return MeshSeries(str(_meshseries_dir / "2D.pvd"))


def load_meshseries_CT_2D_XDMF():
    ms = MeshSeries(str(_meshseries_dir / "elder.xdmf"))
    return ms.transform(lambda mesh: mesh.translate([0, -mesh.center[1], 0]))


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


def load_meshseries_BHE_3D_1P():
    return MeshSeries(_meshseries_dir / "3D_BHE_sandwich" / "sandwich_1P.pvd")


def load_meshseries_BHE_3D_1U():
    return MeshSeries(_meshseries_dir / "3D_BHE_sandwich" / "sandwich_1U.pvd")


def load_meshseries_BHE_3D_2U():
    return MeshSeries(_meshseries_dir / "3D_BHE_sandwich" / "sandwich_2U.pvd")


def load_meshseries_BHE_3D_CXA():
    return MeshSeries(_meshseries_dir / "3D_BHE_sandwich" / "sandwich_CXA.pvd")


def load_meshseries_BHE_3D_CXC():
    return MeshSeries(_meshseries_dir / "3D_BHE_sandwich" / "sandwich_CXC.pvd")


def load_meshseries_BHEs_3D(kind: Literal["full", "line", "lines"], ext: str):
    name = {"full": "3bhes", "line": "3bhes_1", "lines": "3bhes_1_2_3"}[kind]
    if ext == ".xdmf":
        name = "3bhes_" + name

    return MeshSeries((_meshseries_dir / "3D_BHEs" / name).with_suffix(ext))


def load_meshseries_HT_2D_paraview_XMF():
    return MeshSeries(
        str(_meshseries_dir / "2D_single_fracture_HT_2D_single_fracture.xmf")
    )


def load_meshseries_diffusion_3D(Tb=373.15, Ta=293.15, alpha=1e-6):
    timevalues = np.geomspace(1e3, 1e7, num=12)
    xg, yg, zg = (np.linspace(0, 1, num) for num in [40, 2, 2])
    mesh = pv.StructuredGrid(*np.meshgrid(xg, yg, zg, indexing="ij"))
    meshes = []
    x = mesh.points[:, 0]
    for tv in timevalues:
        offset = 2e1 * (1.1 - x) / tv**0.2
        mesh["temperature"] = (
            anasol.heat_conduction_temperature(x, tv, Tb, Ta, alpha) + offset
        )
        meshes += [Mesh(mesh.copy())]
    return MeshSeries.from_data(meshes, timevalues)


def load_meshseries_PETSc_2D():
    return MeshSeries(
        str(_meshseries_dir / "2D_PETSC" / "square_1e1_neumann.pvd")
    )


def load_mesh_mechanics_2D():
    return Mesh.read(_meshseries_dir / "mechanics_2D.vtu")


def load_mesh_mechanics_3D_cylinder():
    # from ogs/Tests/Data/Mechanics/Linear/ElementDeactivation3D/element_deactivation_M_3D.prj
    # Adjusted params to get reasonable stresses
    # clipped manually to 1/8 due to symmetry
    return Mesh.read(_meshseries_dir / "mechanics_3D_cylinder.vtu")


def load_mesh_mechanics_3D_sphere():
    # from ogs/Tests/Data/Mechanics/Linear/PressureBC/hollow_sphere.prj
    # Adjusted params to get reasonable stresses
    return Mesh.read(_meshseries_dir / "mechanics_3D_sphere.vtu")


def load_meshes_liquid_flow_simple():
    # In this example, we create the domain mesh (a 10x2 rectangle) and the boundary meshes for the simulation
    # We also set the boundary condition (prescribed pressure) on the left and right boundary meshes.
    from pathlib import Path

    import ogstools as ot

    workingdir = Path()
    gmsh_file = workingdir / "rect_10_2.msh"

    ot.meshlib.rect(
        lengths=(10, 2),
        n_edge_cells=(10, 4),
        n_layers=2,
        structured_grid=True,
        order=1,
        mixed_elements=False,
        jiggle=0.0,
        out_name=gmsh_file,
    )

    meshes = ot.Meshes.from_gmsh(gmsh_file)

    # Add data array 'pressure' to the left and right meshes boundary meshes
    points_shape = np.shape(meshes["left"].points)
    meshes["left"].point_data["pressure"] = np.full(points_shape[0], 2.9e7)
    meshes["right"].point_data["pressure"] = np.full(points_shape[0], 3e7)

    return meshes


def load_model_liquid_flow_simple():
    from ogstools import Project

    prj_file = EXAMPLES_DIR / "prj" / "SimpleLF.prj"
    prj = Project(prj_file)
    meshes = load_meshes_liquid_flow_simple()

    return prj, meshes


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
log_adaptive_timestepping = _logs_dir / "serial_adaptive_timestepping.log"
info_parallel_1 = _logs_dir / "parallel_1_info.txt"
debug_parallel_3 = _logs_dir / "parallel_3_debug.txt"
serial_convergence_long = _logs_dir / "serial_convergence_long.txt"
serial_convergence_short = _logs_dir / "serial_convergence_short.txt"
serial_critical = _logs_dir / "serial_critical.txt"
serial_info = _logs_dir / "serial_info.txt"
serial_time_step_rejected = _logs_dir / "serial_time_step_rejected.txt"
serial_warning_only = _logs_dir / "serial_warning_only.txt"
serial_v2_coupled_ht = _logs_dir / "serial_v2_coupled_ht.txt"
serial_v2_staggered_ht = _logs_dir / "serial_v2_staggered_ht.txt"
log_petsc_mpi_1 = _logs_dir / "petscMPI1.log"
log_petsc_mpi_2 = _logs_dir / "petscMPI2.log"

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
prj_beier_sandbox_add_output_ref = _prj_dir / "beier_sandbox_add_output_ref.prj"
prj_3bhes_id_1U_2U_1U = _prj_dir / "3bhes_id_1U_2U_1U.prj"
prj_3bhes_id_1U_2U_1U_ref = _prj_dir / "3bhes_id_1U_2U_1U_ref.prj"
prj_beier_sandbox_power_ref = _prj_dir / "beier_sandbox_power_ref.prj"
prj_square_1e4_robin = _prj_dir / "square_1e4_robin.prj"
prj_square_1e4_robin_ref = _prj_dir / "square_1e4_robin_ref.prj"
pybc_nuclear_decay = _prj_dir / "decay_boundary_conditions.py"
prj_th2m_phase_transition = _prj_dir / "th2m_phase_transition.prj"


surface_paths = [
    _surface_dir / (file + ".vtu")
    for file in ["00_KB", "01_q", "02_krl", "03_S3", "04_krp"]
]

test_shapefile = _shp_dir / "test_shape.shp"
circle_shapefile = _shp_dir / "circle.shp"

elder_h5 = _meshseries_dir / "elder.h5"
elder_xdmf = _meshseries_dir / "elder.xdmf"

mechanics_2D = _meshseries_dir / "mechanics_2D.vtu"
example_hlw = _yaml_mesh_dir / "example_hlw.yml"
