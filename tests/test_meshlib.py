"""Unit tests for meshlib."""

import numpy as np
import pytest
from pyvista import SolidSphere, UnstructuredGrid

import ogstools as ot
from ogstools import examples


class TestUtils:
    """Test case for ogstools utilities."""

    def test_all_types(self):
        pvd = examples.load_meshseries_THM_2D_PVD()
        xdmf = examples.load_meshseries_HT_2D_XDMF()
        pytest.raises(TypeError, ot.MeshSeries, __file__)

        for ms in [pvd, xdmf]:
            assert ms.read(0) == ms.read_closest(1e-6)

            assert not np.any(np.isnan(ms.timesteps))
            assert not np.any(np.isnan(ms.values("temperature")))

            assert ms.timevalues[
                ms.closest_timestep(1.0)
            ] == ms.closest_timevalue(1.0)

            ms.clear()

    def test_aggregate(self):
        "Test aggregation of meshseries."
        mesh_series = examples.load_meshseries_HT_2D_XDMF()
        funcs = ["min", "max", "mean", "median", "sum", "std", "var"]
        timefuncs = ["min_time", "max_time"]
        for func in funcs + timefuncs:
            agg_mesh = mesh_series.aggregate("temperature", func)
            assert not np.any(np.isnan(agg_mesh["temperature_" + func]))

    def test_aggregate_mesh_dependent(self):
        "Test aggregation of mesh_dependent property on meshseries."
        mesh_series = examples.load_meshseries_THM_2D_PVD()
        for func in ["max", "max_time"]:
            agg_mesh = mesh_series.aggregate(
                ot.properties.dilatancy_alkan, func
            )
            data_name = ot.properties.dilatancy_alkan.output_name + "_" + func
            assert not np.any(np.isnan(agg_mesh[data_name]))

    def test_probe_pvd(self):
        "Test point probing on pvd."
        mesh_series = examples.load_meshseries_THM_2D_PVD()
        points = mesh_series.read(0).cell_centers().points
        for method in ["nearest", "probefilter"]:
            values = mesh_series.probe(points, "temperature", method)
            assert not np.any(np.isnan(values))

    def test_plot_probe(self):
        """Test creation of probe plots."""
        meshseries = examples.load_meshseries_THM_2D_PVD()
        points = meshseries.read(0).center
        meshseries.plot_probe(points, ot.properties.temperature)
        points = meshseries.read(0).points[[0, -1]]
        meshseries.plot_probe(points, ot.properties.temperature)
        meshseries.plot_probe(points, ot.properties.velocity)
        meshseries.plot_probe(points, ot.properties.stress)
        meshseries.plot_probe(points, ot.properties.stress.von_Mises)
        mesh_series = examples.load_meshseries_HT_2D_XDMF()
        points = mesh_series.read(0).center
        meshseries.plot_probe(points, ot.properties.temperature)
        meshseries.plot_probe(points, ot.properties.velocity)

    def test_probe_xdmf(self):
        "Test point probing on xdmf."
        mesh_series = examples.load_meshseries_HT_2D_XDMF()
        points = mesh_series.read(0).cell_centers().points
        for method in ["nearest", "linear", None]:
            values = mesh_series.probe(points, "temperature", method)
            assert not np.any(np.isnan(values))

    def test_diff_two_meshes(self):
        meshseries = examples.load_meshseries_THM_2D_PVD()
        mesh1 = meshseries.read(0)
        mesh2 = meshseries.read(-1)
        mesh_diff = ot.meshlib.difference(mesh1, mesh2, "temperature")
        mesh_diff = ot.meshlib.difference(
            mesh1, mesh2, ot.properties.temperature
        )
        assert isinstance(mesh_diff, UnstructuredGrid)
        mesh_diff = ot.meshlib.difference(mesh1, mesh2)

    def test_diff_pairwise(self):
        n = 5
        meshseries = examples.load_meshseries_THM_2D_PVD()
        meshes1 = [meshseries.read(0)] * n
        meshes2 = [meshseries.read(-1)] * n
        meshes_diff = ot.meshlib.difference_pairwise(
            meshes1, meshes2, ot.properties.temperature
        )
        assert isinstance(meshes_diff, np.ndarray)
        assert len(meshes_diff) == n

        meshes_diff = ot.meshlib.difference_pairwise(meshes1, meshes2)

    def test_diff_matrix_single(self):
        meshseries = examples.load_meshseries_THM_2D_PVD()
        meshes1 = [meshseries.read(0), meshseries.read(-1)]
        meshes_diff = ot.meshlib.difference_matrix(
            meshes1, mesh_property=ot.properties.temperature
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
            meshes1, meshes2, ot.properties.temperature
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
        self.assertTrue(np.all(mesh["depth"] == -mesh.points[..., 1]))
        mesh["depth"] = mesh.depth()
        self.assertTrue(np.all(mesh["depth"] < -mesh.points[..., 1]))

    def test_depth_3D(self):
        mesh = ot.Mesh(SolidSphere(100, center=(0, 0, -101)))
        mesh["depth"] = mesh.depth(use_coords=True)
        self.assertTrue(np.all(mesh["depth"] == -mesh.points[..., -1]))
        mesh["depth"] = mesh.depth()
        self.assertTrue(np.all(mesh["depth"] < -mesh.points[..., -1]))

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
            ot.properties.temperature,
            profile,
            resolution=10,
        )
        data = ms_sp[ot.properties.temperature.data_name].to_numpy()
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
