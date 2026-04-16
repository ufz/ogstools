"""Unit tests for meshlib."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

import ogstools as ot
from ogstools import examples
from ogstools.examples import prj_mechanics


@pytest.mark.tools  # ipDataToPointCloud
@pytest.mark.parametrize(
    "mesh_load",
    [
        examples.load_mesh_mechanics_2D,
        lambda: examples.load_meshseries_THM_2D_PVD()[-1],
        lambda: examples.load_simulation_smalldeformation().meshseries[-1],
    ],
)
def test_ip_data_init(mesh_load):
    ip_data = ot.mesh.IPdata(mesh_load())
    assert ip_data.n_points != 0


@pytest.mark.tools  # ipDataToPointCloud
@pytest.mark.parametrize(("n_comps", "val"), [(1, 0.0), (4, [1, 1, 1, 0])])
def test_set_ip_data(n_comps: int, val):
    mesh = examples.load_mesh_mechanics_2D()
    ip_data = ot.mesh.IPdata(mesh)
    values = np.full((ip_data.n_points, n_comps), val).squeeze()
    prev_keys = list(mesh.field_data.keys())
    # test setting entire array
    ip_data.set("new_data", order=2, num_components=n_comps, values=values)
    # test broadcasting
    ip_data.set("new_data", order=2, num_components=n_comps, values=val)

    np.testing.assert_array_equal(mesh.field_data["new_data"], values)
    ip_meta = ot.mesh.ip_metadata(mesh)
    assert ip_meta is not None
    names = [arr["name"] for arr in ip_meta["integration_point_arrays"]]
    assert "new_data" in names
    assert set(mesh.field_data.keys()) == set(prev_keys + ["new_data"])
    assert len(mesh.field_data) == len(prev_keys) + 1


def test_delete():
    mesh = examples.load_mesh_mechanics_2D()
    prev_len = len(mesh.field_data)
    ip_data = ot.mesh.IPdata(mesh)
    del ip_data["epsilon_ip"]
    ip_meta = ot.mesh.ip_metadata(mesh)
    names = [arr["name"] for arr in ip_meta["integration_point_arrays"]]
    assert "epsilon_ip" not in names
    assert len(mesh.field_data) == prev_len - 1


def test_modify_simple():
    mesh = examples.load_mesh_mechanics_2D()
    ip_data = ot.mesh.IPdata(mesh)
    ip_data["epsilon_ip"].values *= 0
    np.testing.assert_array_equal(mesh.field_data["epsilon_ip"], 0.0)


@pytest.mark.tools  # ipDataToPointCloud
@pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 30})
def test_modify_complex() -> plt.Figure:
    mesh = examples.load_mesh_mechanics_2D()
    ip_data = ot.mesh.IPdata(mesh)
    ip_mesh = ot.mesh.to_ip_mesh(mesh)
    pts = ip_mesh.cell_centers().points
    mask = np.hypot(pts[:, 0] - 150, pts[:, 1] + 650) < 90
    ip_data["sigma_ip"].values[mask, :3] -= 2.1e6

    sigma_ip = ot.variables.stress.replace(
        data_name="sigma_ip", output_name="IP_stress"
    )
    return ot.plot.contourf(ot.mesh.to_ip_mesh(mesh), sigma_ip.trace)


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.system
def test_to_ip_mesh_mixed(tmp_path, mixed):
    "Test to_ip_mesh works with single and mixed cell types."
    rect = ot.gmsh_tools.rect(
        n_edge_cells=6,
        n_layers=2,
        structured_grid=True,
        mixed_elements=mixed,
    )
    meshes = ot.Meshes.from_gmsh(rect)
    prj = ot.Project(input_file=prj_mechanics).copy()
    model = ot.Model(prj, meshes)
    model._next_target = tmp_path  # use only in testing!
    sim = model.run()
    assert sim.status == sim.Status.done, f"Simulation status: {sim.status_str}"
    mesh_last = sim.meshseries[-1]
    n_cell_types = len(np.unique(mesh_last.celltypes))
    assert n_cell_types == (
        2 if mixed else 1
    ), f"Expected {'mixed' if mixed else 'uniform'} cell types, got {n_cell_types}"
    int_pts = ot.mesh.to_ip_point_cloud(mesh_last)
    ip_mesh = ot.mesh.to_ip_mesh(mesh_last)
    assert int_pts.number_of_points == ip_mesh.number_of_cells


@pytest.mark.tools  # ipDataToPointCloud
@pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 30})
def test_modify_material() -> plt.Figure:
    mesh = examples.load_mesh_mechanics_2D()
    ip_data = ot.mesh.IPdata(mesh)
    ip_cloud = ot.mesh.to_ip_point_cloud(mesh)
    cell_map = mesh.find_containing_cell(ip_cloud.points)
    ip_mat_ids = mesh["MaterialIDs"][cell_map]
    ip_data["sigma_ip"].values[ip_mat_ids == 0, 0] -= 10e6

    sigma_ip = ot.variables.stress.replace(
        data_name="sigma_ip", output_name="IP_stress"
    )
    return ot.plot.contourf(ot.mesh.to_ip_mesh(mesh), sigma_ip.trace)
