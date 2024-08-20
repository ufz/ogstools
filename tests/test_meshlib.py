"""Unit tests for meshlib."""

from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import pytest
from ogs6py import ogs
from pyvista import SolidSphere, UnstructuredGrid

import ogstools as ot
from ogstools import examples
from ogstools.msh2vtu import msh2vtu


class TestUtils:
    """Test case for ogstools utilities."""

    def test_meshseries_xdmf(self):
        # all data is in one group in one h5 file
        xdmf = examples.load_meshseries_HT_2D_XDMF()
        # all data is in separated groups of one h5 file
        xmf = examples.load_meshseries_HT_2D_paraview_XMF()

        for ht in [xdmf, xmf]:
            # temperature is scalar last dimension = 1 omitted
            assert np.shape(ht.values("temperature")) == (97, 190)
            # non-scalar attribute
            assert np.shape(ht.values("darcy_velocity")) == (97, 190, 2)

            assert np.shape(ht.select("temperature")[1:3, :]) == (2, 190)
            # no range for a dimension -> this dimension gets omitted
            assert np.shape(ht.select("temperature")[1, :]) == (190,)
            # select range with length for a dimension to keep dimension
            assert np.shape(ht.select("temperature")[1:2, :]) == (1, 190)
            # all values
            assert np.shape(ht.select("temperature")[:]) == np.shape(
                ht.values("temperature")
            )
            # last 2 steps
            last_2_steps = ht.select("darcy_velocity")[-2:, 1:4, :]
            assert np.shape(last_2_steps) == (2, 3, 2)

    def test_all_types(self):
        pvd = examples.load_meshseries_THM_2D_PVD()
        xdmf = examples.load_meshseries_HT_2D_XDMF()
        with pytest.raises(TypeError):
            ot.MeshSeries(
                __file__, match="Can only read 'pvd', 'xdmf' or 'vtu' files."
            )

        for ms in [pvd, xdmf]:
            try:
                mesh1 = ms.read(0)
                mesh1_closest = ms.read_closest(1e-6)
            except Exception:
                pytest.fail("Read functions of MeshSeries failed")

            assert mesh1 == mesh1_closest
            assert not np.any(np.isnan(ms.timesteps))
            assert not np.any(np.isnan(ms.values("temperature")))

            assert ms.timevalues()[
                ms.closest_timestep(1.0)
            ] == ms.closest_timevalue(1.0)

            ms.clear()

    def test_time_aggregate(self):
        "Test aggregation of meshseries."
        mesh_series = examples.load_meshseries_HT_2D_XDMF()
        funcs = ["min", "max", "mean", "median", "sum", "std", "var"]
        for func in funcs:
            agg_mesh = mesh_series.aggregate_over_time("temperature", func)
            assert not np.any(np.isnan(agg_mesh["temperature_" + func]))

    def test_aggregate_time_of_limit(self):
        "Test aggregation of meshseries."
        mesh_series = examples.load_meshseries_HT_2D_XDMF()
        min_mesh = mesh_series.time_of_min("temperature")
        assert not np.any(np.isnan(min_mesh["min_temperature_time"]))
        max_mesh = mesh_series.time_of_max("temperature")
        assert not np.any(np.isnan(max_mesh["max_temperature_time"]))

    def test_time_aggregate_mesh_dependent(self):
        "Test aggregation of mesh_dependent variable on meshseries."
        mesh_series = examples.load_meshseries_THM_2D_PVD()
        prop = ot.variables.dilatancy_alkan
        agg_mesh = mesh_series.aggregate_over_time(prop, "max")
        assert not np.any(np.isnan(agg_mesh[prop.output_name + "_max"]))
        agg_mesh = mesh_series.time_of_max(prop)
        assert not np.any(np.isnan(agg_mesh[f"max_{prop.output_name}_time"]))

    def test_domain_aggregate(self):
        "Test aggregation of meshseries."
        mesh_series = examples.load_meshseries_THM_2D_PVD()
        mask = mesh_series.read(0).ctp()["MaterialIDs"] > 3
        values = mesh_series.aggregate_over_domain(
            "temperature", "max", mask=mask
        )
        assert not np.any(np.isnan(values))

    def test_plot_domain_aggregate(self):
        "Test aggregation of meshseries."
        mesh_series = examples.load_meshseries_THM_2D_PVD()
        mesh_series.plot_domain_aggregate(
            "temperature", "mean", slice(1, -1), "a"
        )

    def test_time_slice(self):
        mesh_series = examples.load_meshseries_HT_2D_XDMF()
        points = np.linspace([2, 2, 0], [4, 18, 0], num=100)
        mesh_series.plot_time_slice("temperature", points, levels=[78, 79, 80])
        mesh_series.plot_time_slice(
            "temperature", points, y_axis="y", interpolate=False, time_unit="h",
            time_logscale=True, cb_loc="left", dpi=50, fontsize=10
        )  # fmt: skip

    def test_probe(self):
        "Test point probing on meshseries."
        mesh_series = examples.load_meshseries_HT_2D_XDMF()
        points = mesh_series.read(0).cell_centers().points
        for method in ["nearest", "linear"]:
            values = mesh_series.probe(points, "temperature", method)
            assert not np.any(np.isnan(values))

    def test_plot_probe(self):
        """Test creation of probe plots."""
        meshseries = examples.load_meshseries_THM_2D_PVD()
        points = meshseries.read(0).center
        meshseries.plot_probe(points, ot.variables.temperature)
        points = meshseries.read(0).points[[0, -1]]
        meshseries.plot_probe(points, ot.variables.temperature)
        meshseries.plot_probe(points, ot.variables.velocity)
        meshseries.plot_probe(points, ot.variables.stress)
        meshseries.plot_probe(points, ot.variables.stress.von_Mises)
        mesh_series = examples.load_meshseries_HT_2D_XDMF()
        points = mesh_series.read(0).center
        meshseries.plot_probe(points, ot.variables.temperature)
        meshseries.plot_probe(points, ot.variables.velocity)

    def test_diff_two_meshes(self):
        meshseries = examples.load_meshseries_THM_2D_PVD()
        mesh1 = meshseries.read(0)
        mesh2 = meshseries.read(-1)
        mesh_diff = ot.meshlib.difference(mesh1, mesh2, "temperature")
        mesh_diff = ot.meshlib.difference(
            mesh1, mesh2, ot.variables.temperature
        )
        assert isinstance(mesh_diff, UnstructuredGrid)
        mesh_diff = ot.meshlib.difference(mesh1, mesh2)

    def test_diff_pairwise(self):
        n = 5
        meshseries = examples.load_meshseries_THM_2D_PVD()
        meshes1 = [meshseries.read(0)] * n
        meshes2 = [meshseries.read(-1)] * n
        meshes_diff = ot.meshlib.difference_pairwise(
            meshes1, meshes2, ot.variables.temperature
        )
        assert isinstance(meshes_diff, np.ndarray)
        assert len(meshes_diff) == n

        meshes_diff = ot.meshlib.difference_pairwise(meshes1, meshes2)

    def test_diff_matrix_single(self):
        meshseries = examples.load_meshseries_THM_2D_PVD()
        meshes1 = [meshseries.read(0), meshseries.read(-1)]
        meshes_diff = ot.meshlib.difference_matrix(
            meshes1, variable=ot.variables.temperature
        )
        assert isinstance(meshes_diff, np.ndarray)

        assert meshes_diff.shape == (
            len(meshes1),
            len(meshes1),
        )

        meshes_diff = ot.meshlib.difference_matrix(meshes1)

    def test_diff_matrix_unequal(self):
        meshseries = examples.load_meshseries_THM_2D_PVD()
        meshes1 = [meshseries.read(0), meshseries.read(-1)]
        meshes2 = [meshseries.read(0), meshseries.read(-1), meshseries.read(-1)]
        meshes_diff = ot.meshlib.difference_matrix(
            meshes1, meshes2, ot.variables.temperature
        )
        assert isinstance(meshes_diff, np.ndarray)
        assert meshes_diff.shape == (
            len(meshes1),
            len(meshes2),
        )
        meshes_diff = ot.meshlib.difference_matrix(meshes1, meshes2)

    def test_depth_2D(self):
        mesh = examples.load_mesh_mechanics_2D()
        mesh["depth"] = mesh.depth(use_coords=True)
        # y Axis is vertical axis
        assert np.all(mesh["depth"] == -mesh.points[..., 1])
        mesh["depth"] = mesh.depth()
        assert np.all(mesh["depth"] < -mesh.points[..., 1])

    def test_depth_3D(self):
        mesh = ot.Mesh(SolidSphere(100, center=(0, 0, -101)))
        mesh["depth"] = mesh.depth(use_coords=True)
        assert np.all(mesh["depth"] == -mesh.points[..., -1])
        mesh["depth"] = mesh.depth()
        assert np.all(mesh["depth"] < -mesh.points[..., -1])

    def test_interp_points(self):
        profile = np.array(
            [
                [-1000, -175, 6700],
                [-600, -600, 6700],
                [100, -300, 6700],
            ]
        )
        profile_points = ot.meshlib.interp_points(profile, resolution=100)
        assert isinstance(profile_points, np.ndarray)
        # Check first point
        assert (profile_points[0, :] == profile[0, :]).all()
        # Check last point
        assert (profile_points[-1, :] == profile[2, :]).all()
        # Check if middle point is present in the profile at expected index
        assert (profile_points == profile[1, :]).any()

    def test_distance_in_segments(self):
        profile = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        profile_points = ot.meshlib.interp_points(profile, resolution=9)
        dist_in_seg = ot.meshlib.distance_in_segments(profile, profile_points)
        assert len(np.where(dist_in_seg == 0)[0]) == 2
        assert len(np.where(dist_in_seg == 1)[0]) == 1

    def test_distance_in_profile(self):
        profile = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        profile_points = ot.meshlib.interp_points(profile, resolution=9)
        dist_in_seg = ot.meshlib.distance_in_profile(profile_points)
        # Check if distance is increasing
        assert np.all(np.diff(dist_in_seg) > 0)
        # Check if distances at the beginning and end of profile are correct
        assert dist_in_seg[0] == 0
        assert dist_in_seg[-1] == 2

    def test_sample_over_polyline_single_segment(self):
        ms = examples.load_meshseries_HT_2D_XDMF()
        profile = np.array([[4, 2, 0], [4, 18, 0]])
        ms_sp, _ = ot.meshlib.sample_polyline(
            ms.read(-1),
            ["pressure", "temperature"],
            profile,
        )
        assert not (np.any(np.isnan(ms_sp["pressure"])))
        assert (ms_sp["pressure"].to_numpy() > 0).all()
        assert not (np.any(np.isnan(ms_sp["temperature"])))
        assert (
            ms_sp["temperature"].to_numpy()
            >= np.zeros_like(ms_sp["temperature"].to_numpy())
        ).all()
        assert (ms_sp["dist"] == ms_sp["dist_in_segment"]).all()

    def test_sample_over_polyline_multi_segment(self):
        ms = examples.load_meshseries_THM_2D_PVD()
        profile = np.array(
            [
                [-1000, -175, 6700],
                [-600, -600, 6700],
                [100, -300, 6700],
                [910, -590, 6700],
            ]
        )
        ms_sp, _ = ot.meshlib.sample_polyline(
            ms.read(1),
            ot.variables.temperature,
            profile,
            resolution=10,
        )
        data = ms_sp[ot.variables.temperature.data_name].to_numpy()
        assert not np.any(np.isnan(data))
        assert (np.abs(data) > np.zeros_like(data)).all()
        # output should be in Celsius
        assert (data >= np.ones_like(data) * -273.15).all()
        assert (ms_sp["dist"] != ms_sp["dist_in_segment"]).any()

    def test_sample_over_polyline_single_segment_vec_prop(self):
        ms = examples.load_meshseries_HT_2D_XDMF()
        profile = np.array([[4, 2, 0], [4, 18, 0]])
        ms_sp, _ = ot.meshlib.sample_polyline(
            ms.read(-1),
            "darcy_velocity",
            profile,
        )
        assert not np.any(np.isnan(ms_sp["darcy_velocity_0"]))
        assert (
            np.abs(ms_sp["darcy_velocity_0"].to_numpy())
            > np.zeros_like(ms_sp["darcy_velocity_0"].to_numpy())
        ).all()
        assert not np.any(np.isnan(ms_sp["darcy_velocity_1"]))
        assert (
            np.abs(ms_sp["darcy_velocity_1"].to_numpy())
            > np.zeros_like(ms_sp["darcy_velocity_1"].to_numpy())
        ).all()

    def test_ip_mesh(self):
        "Test creation of integration point meshes."

        tmp_path = Path(mkdtemp())
        mesh_path = Path(tmp_path) / "mesh.msh"
        sigma_ip = ot.variables.stress.replace(data_name="sigma_ip")

        def run_and_check(
            elem_order: int, quads: bool, intpt_order: int, mixed: bool = False
        ):
            ot.meshlib.gmsh_meshing.rect(
                n_edge_cells=6,
                n_layers=2,
                structured_grid=quads,
                order=elem_order,
                out_name=mesh_path,
                mixed_elements=mixed,
                jiggle=0.01,
            )
            msh2vtu(mesh_path, tmp_path, reindex=True, log_level="ERROR")

            model = ogs.OGS(
                PROJECT_FILE=tmp_path / "default.prj",
                INPUT_FILE=examples.prj_mechanics,
            )
            model.replace_text(intpt_order, xpath=".//integration_order")
            model.write_input()
            model.run_model(
                write_logs=True, args=f"-m {tmp_path} -o {tmp_path}"
            )
            meshseries = ot.MeshSeries(tmp_path / "mesh.pvd")
            int_pts = meshseries.read(-1).to_ip_point_cloud()
            ip_ms = meshseries.ip_tesselated()
            ip_mesh = ip_ms.read(-1)
            vals = ip_ms.probe(ip_mesh.center, sigma_ip.data_name)
            assert not np.any(np.isnan(vals))
            assert int_pts.number_of_points == ip_mesh.number_of_cells
            containing_cells = ip_mesh.find_containing_cell(int_pts.points)
            # check for integration points coinciding with the tessellated cells
            np.testing.assert_equal(
                sigma_ip.magnitude.transform(ip_mesh)[containing_cells],
                sigma_ip.magnitude.transform(int_pts),
            )

        run_and_check(elem_order=1, quads=False, intpt_order=2)
        run_and_check(elem_order=1, quads=False, intpt_order=3)
        run_and_check(elem_order=1, quads=False, intpt_order=4)
        run_and_check(elem_order=2, quads=False, intpt_order=4)
        run_and_check(elem_order=1, quads=True, intpt_order=2)
        run_and_check(elem_order=1, quads=True, intpt_order=3)
        run_and_check(elem_order=1, quads=True, intpt_order=4)
        run_and_check(elem_order=2, quads=True, intpt_order=4)
        run_and_check(elem_order=2, quads=True, intpt_order=4)
        run_and_check(elem_order=1, quads=False, intpt_order=2, mixed=True)

    def test_reader(self):
        assert isinstance(examples.load_meshseries_THM_2D_PVD(), ot.MeshSeries)
        assert isinstance(ot.MeshSeries(examples.elder_xdmf), ot.MeshSeries)
        assert isinstance(ot.Mesh.read(examples.mechanics_vtu), ot.Mesh)
        assert isinstance(ot.Mesh.read(examples.test_shapefile), ot.Mesh)

    def test_xdmf_quadratic(self):
        "Test reading of quadratic elements in xdmf."

        tmp_path = Path(mkdtemp())
        mesh_path = Path(tmp_path) / "mesh.msh"
        ot.meshlib.gmsh_meshing.rect(
            n_edge_cells=6, structured_grid=False, order=2, out_name=mesh_path
        )
        msh2vtu(mesh_path, tmp_path, reindex=True, log_level="ERROR")
        model = ogs.OGS(
            PROJECT_FILE=tmp_path / "default.prj",
            INPUT_FILE=examples.prj_mechanics,
        )
        model.replace_text("4", xpath=".//integration_order")
        model.replace_text("XDMF", xpath="./time_loop/output/type")
        model.write_input()
        model.run_model(write_logs=True, args=f"-m {tmp_path} -o {tmp_path}")
        mesh = ot.MeshSeries(tmp_path / "mesh_mesh_domain.xdmf").read(-1)
        assert not np.any(np.isnan(ot.variables.stress.transform(mesh)))

    def test_remesh_with_tri(self):
        mesh = examples.load_meshseries_THM_2D_PVD().read(1)
        temp_dir = Path(mkdtemp())
        msh_path = temp_dir / "tri_mesh.msh"
        ot.meshlib.gmsh_meshing.remesh_with_triangle(mesh, msh_path)
        assert (
            msh2vtu(msh_path, temp_dir, reindex=False, log_level="ERROR") == 0
        )
