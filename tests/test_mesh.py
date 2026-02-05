"""Unit tests for meshlib."""

from pathlib import Path

import numpy as np
import pytest
import pyvista as pv
from hypothesis import given, settings
from hypothesis import strategies as st

import ogstools as ot
from ogstools import examples
from ogstools.mesh import utils


def test_diff_two_meshes():
    meshseries = examples.load_meshseries_THM_2D_PVD()
    mesh1 = meshseries.mesh(0)
    mesh2 = meshseries.mesh(-1)
    mesh_diff = ot.mesh.difference(mesh1, mesh2, "temperature")
    # test, that no sampling occurs for equal topology
    np.testing.assert_array_equal(
        mesh_diff["temperature_difference"],
        mesh1["temperature"] - mesh2["temperature"],
    )
    # test same/different topology and scalar / vector variable
    for scaling in [1.0, 2.0]:
        for variable in ["temperature", "velocity"]:
            mesh_diff = ot.mesh.difference(
                mesh1.scale(scaling), mesh2, variable
            )

    quad_tri_diff = ot.mesh.difference(
        mesh1.triangulate(), mesh1, "temperature"
    )
    quad_tri_diff_vals = ot.variables.temperature.difference.transform(
        quad_tri_diff
    )
    np.testing.assert_allclose(quad_tri_diff_vals, 0.0, atol=1e-12)
    mesh_diff = ot.mesh.difference(mesh1, mesh2, ot.variables.temperature)
    assert isinstance(mesh_diff, pv.UnstructuredGrid)
    mesh_diff = ot.mesh.difference(mesh1, mesh2)


def test_diff_pairwise():
    n = 5
    meshseries = examples.load_meshseries_THM_2D_PVD()
    meshes1 = [meshseries.mesh(0)] * n
    meshes2 = [meshseries.mesh(-1)] * n
    meshes_diff = ot.mesh.difference_pairwise(
        meshes1, meshes2, ot.variables.temperature
    )
    assert isinstance(meshes_diff, np.ndarray)
    assert len(meshes_diff) == n

    meshes_diff = ot.mesh.difference_pairwise(meshes1, meshes2)


def test_diff_matrix_single():
    meshseries = examples.load_meshseries_THM_2D_PVD()
    meshes1 = [meshseries.mesh(0), meshseries.mesh(-1)]
    meshes_diff = ot.mesh.difference_matrix(
        meshes1, variable=ot.variables.temperature
    )
    assert isinstance(meshes_diff, np.ndarray)

    assert meshes_diff.shape == (
        len(meshes1),
        len(meshes1),
    )

    meshes_diff = ot.mesh.difference_matrix(meshes1)


def test_diff_matrix_unequal():
    meshseries = examples.load_meshseries_THM_2D_PVD()
    meshes1 = [meshseries.mesh(0), meshseries.mesh(-1)]
    meshes2 = [meshseries.mesh(0), meshseries.mesh(-1), meshseries.mesh(-1)]
    meshes_diff = ot.mesh.difference_matrix(
        meshes1, meshes2, ot.variables.temperature
    )
    assert isinstance(meshes_diff, np.ndarray)
    assert meshes_diff.shape == (
        len(meshes1),
        len(meshes2),
    )
    meshes_diff = ot.mesh.difference_matrix(meshes1, meshes2)


def test_depth_2D():
    mesh = examples.load_mesh_mechanics_2D()
    top_mesh = ot.Meshes.from_mesh(mesh)["top"]
    depth = ot.mesh.depth(mesh, top_mesh)
    assert np.isclose(np.ptp(depth), np.ptp(mesh.points[:, 1]))


def test_depth_3D():
    mesh = pv.SolidSphere(100, center=(0, 0, -101))
    top_mesh = mesh.extract_surface().clip("-z")
    depth = ot.mesh.depth(mesh, top_mesh)
    assert np.isclose(np.ptp(depth), np.ptp(mesh.points[:, 2]))


def test_reshape_obs_points():
    points_x = (1,)
    pts_x = utils.reshape_obs_points(points_x)
    assert pts_x.shape == (1, 3)


def test_reshape_obs_points_mesh():
    ms = examples.load_meshseries_CT_2D_XDMF()
    mesh = ms.mesh(0)
    points = ((-150.0, 75.0), (-147.65625, 72.65625))
    pts = utils.reshape_obs_points(points, mesh)
    np.testing.assert_equal(
        pts, np.asarray([[-150, 0, 75], [-147.65625, 0, 72.65625]])
    )


@pytest.mark.tools()
@settings(max_examples=30, deadline=1600)
@given(
    st.one_of(
        st.integers(2, 13),
        st.tuples(st.integers(1, 14), st.integers(1, 14))
        .filter(lambda x: abs(x[1] - x[0]) < 13)
        .map(sorted),
    ),
    st.booleans(),
)
def test_threshold_ip_data(mat_ids: tuple, invert: bool):
    "Check length of thresholded ip data is correct."
    mesh = examples.load_meshseries_THM_2D_PVD()[-1]
    thresh_ip_data = ot.mesh.ip_data_threshold(mesh, mat_ids, invert=invert)
    thresh_n_cells = mesh.threshold(
        mat_ids, scalars="MaterialIDs", invert=invert
    ).n_cells

    for arr in ot.mesh.ip_metadata(mesh):
        n_ip = len(mesh[arr["name"]]) // mesh.n_cells
        assert len(thresh_ip_data[arr["name"]]) == (thresh_n_cells * n_ip)


@pytest.mark.parametrize("strict", [True, False])
@pytest.mark.parametrize(
    "mesh",
    [
        examples.load_mesh_mechanics_2D(),
        examples.load_mesh_mechanics_3D_cylinder(),
        examples.load_meshseries_THM_2D_PVD()[0],
    ],
)
def test_mesh_validate(mesh: pv.UnstructuredGrid | Path, strict: bool):
    assert ot.mesh.validate(mesh, strict=strict)
    # intentionally reversing the node order with method 0
    wrong_mesh = mesh.copy().extract_surface().flip_faces()
    if strict:
        with pytest.raises(UserWarning, match="not compliant with OGS"):
            ot.mesh.validate(wrong_mesh, strict=strict)
    else:
        assert not ot.mesh.validate(wrong_mesh, strict=strict)
