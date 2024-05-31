"""Unit tests for meshlib."""

import unittest

import numpy as np
from pyvista import UnstructuredGrid

from ogstools import examples
from ogstools.meshlib import (
    MeshSeries,
    difference,
    difference_matrix,
    difference_pairwise,
    distance_in_profile,
    distance_in_segments,
    interp_points,
    sample_polyline,
)
from ogstools.propertylib import properties


class UtilsTest(unittest.TestCase):
    """Test case for ogstools utilities."""

    def test_all_types(self):
        pvd = examples.load_meshseries_THM_2D_PVD()
        xdmf = examples.load_meshseries_HT_2D_XDMF()
        self.assertRaises(TypeError, MeshSeries, __file__)

        for mesh_series in [pvd, xdmf]:
            self.assertTrue(
                mesh_series.read(0) == mesh_series.read_closest(1e-6)
            )
            self.assertTrue(not np.any(np.isnan(mesh_series.timesteps)))
            self.assertTrue(
                not np.any(np.isnan(mesh_series.values("temperature")))
            )
            self.assertTrue(
                mesh_series.timevalues[mesh_series.closest_timestep(1.0)]
                == mesh_series.closest_timevalue(1.0)
            )
            mesh_series.clear()

    def test_aggregate(self):
        "Test aggregation of meshseries."
        mesh_series = examples.load_meshseries_HT_2D_XDMF()
        funcs = ["min", "max", "mean", "median", "sum", "std", "var"]
        timefuncs = ["min_time", "max_time"]
        for func in funcs + timefuncs:
            agg_mesh = mesh_series.aggregate("temperature", func)
            self.assertTrue(
                not np.any(np.isnan(agg_mesh["temperature_" + func]))
            )

    def test_aggregate_mesh_dependent(self):
        "Test aggregation of mesh_dependent property on meshseries."
        mesh_series = examples.load_meshseries_THM_2D_PVD()
        for func in ["max", "max_time"]:
            agg_mesh = mesh_series.aggregate(properties.dilatancy_alkan, func)
            self.assertTrue(
                not np.any(
                    np.isnan(
                        agg_mesh[
                            properties.dilatancy_alkan.output_name + "_" + func
                        ]
                    )
                )
            )

    def test_probe_pvd(self):
        "Test point probing on pvd."
        mesh_series = examples.load_meshseries_THM_2D_PVD()
        points = mesh_series.read(0).cell_centers().points
        for method in ["nearest", "probefilter"]:
            values = mesh_series.probe(points, "temperature", method)
            self.assertTrue(not np.any(np.isnan(values)))

    def test_probe_xdmf(self):
        "Test point probing on xdmf."
        mesh_series = examples.load_meshseries_HT_2D_XDMF()
        points = mesh_series.read(0).cell_centers().points
        for method in ["nearest", "linear", None]:
            values = mesh_series.probe(points, "temperature", method)
            self.assertTrue(not np.any(np.isnan(values)))

    def test_diff_two_meshes(self):
        meshseries = examples.load_meshseries_THM_2D_PVD()
        mesh1 = meshseries.read(0)
        mesh2 = meshseries.read(-1)
        mesh_diff = difference(mesh1, mesh2, "temperature")
        mesh_diff = difference(mesh1, mesh2, properties.temperature)
        self.assertTrue(isinstance(mesh_diff, UnstructuredGrid))
        mesh_diff = difference(mesh1, mesh2)

    def test_diff_pairwise(self):
        n = 5
        meshseries = examples.load_meshseries_THM_2D_PVD()
        meshes1 = [meshseries.read(0)] * n
        meshes2 = [meshseries.read(-1)] * n
        meshes_diff = difference_pairwise(
            meshes1, meshes2, properties.temperature
        )
        self.assertTrue(
            isinstance(meshes_diff, np.ndarray) and len(meshes_diff) == n
        )
        meshes_diff = difference_pairwise(meshes1, meshes2)

    def test_diff_matrix_single(self):
        meshseries = examples.load_meshseries_THM_2D_PVD()
        meshes1 = [meshseries.read(0), meshseries.read(-1)]
        meshes_diff = difference_matrix(
            meshes1, mesh_property=properties.temperature
        )
        self.assertTrue(
            isinstance(meshes_diff, np.ndarray)
            and meshes_diff.shape == (len(meshes1), len(meshes1))
        )
        meshes_diff = difference_matrix(meshes1)

    def test_diff_matrix_unequal(self):
        meshseries = examples.load_meshseries_THM_2D_PVD()
        meshes1 = [meshseries.read(0), meshseries.read(-1)]
        meshes2 = [meshseries.read(0), meshseries.read(-1), meshseries.read(-1)]
        meshes_diff = difference_matrix(
            meshes1, meshes2, properties.temperature
        )
        self.assertTrue(
            isinstance(meshes_diff, np.ndarray)
            and meshes_diff.shape == (len(meshes1), len(meshes2))
        )
        meshes_diff = difference_matrix(meshes1, meshes2)

    def test_interp_points(self):
        profile = np.array(
            [
                [-1000, -175, 6700],
                [-600, -600, 6700],
                [100, -300, 6700],
            ]
        )
        profile_points = interp_points(profile, resolution=100)
        self.assertTrue(isinstance(profile_points, np.ndarray))
        # Check first point
        self.assertTrue((profile_points[0, :] == profile[0, :]).all())
        # Check last point
        self.assertTrue((profile_points[-1, :] == profile[2, :]).all())
        # Check if middle point is present in the profile at expected index
        self.assertTrue((profile_points == profile[1, :]).any())

    def test_distance_in_segments(self):
        profile = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        profile_points = interp_points(profile, resolution=9)
        dist_in_seg = distance_in_segments(profile, profile_points)
        self.assertTrue(len(np.where(dist_in_seg == 0)[0]) == 2)
        self.assertTrue(len(np.where(dist_in_seg == 1)[0]) == 1)

    def test_distance_in_profile(self):
        profile = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        profile_points = interp_points(profile, resolution=9)
        dist_in_seg = distance_in_profile(profile_points)
        # Check if distance is increasing
        self.assertTrue(np.all(np.diff(dist_in_seg) > 0))
        # Check if distances at the beginning and end of profile are correct
        self.assertTrue(dist_in_seg[0] == 0)
        self.assertTrue(dist_in_seg[-1] == 2)

    def test_sample_over_polyline_single_segment(self):
        ms = examples.load_meshseries_HT_2D_XDMF()
        profile = np.array([[4, 2, 0], [4, 18, 0]])
        ms_sp, _ = sample_polyline(
            ms.read(-1),
            ["pressure", "temperature"],
            profile,
        )
        self.assertTrue(not np.any(np.isnan(ms_sp["pressure"])))
        self.assertTrue((ms_sp["pressure"].to_numpy() > 0).all())
        self.assertTrue(not np.any(np.isnan(ms_sp["temperature"])))
        self.assertTrue(
            (
                ms_sp["temperature"].to_numpy()
                >= np.zeros_like(ms_sp["temperature"].to_numpy())
            ).all(),
        )
        self.assertTrue((ms_sp["dist"] == ms_sp["dist_in_segment"]).all())

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
        ms_sp, _ = sample_polyline(
            ms.read(1),
            properties.temperature,
            profile,
            resolution=10,
        )
        self.assertTrue(
            not np.any(np.isnan(ms_sp[properties.temperature.data_name]))
        )
        self.assertTrue(
            (
                np.abs(ms_sp[properties.temperature.data_name].to_numpy())
                > np.zeros_like(
                    ms_sp[properties.temperature.data_name].to_numpy()
                )
            ).all()
        )
        self.assertTrue(
            (
                ms_sp["temperature"].to_numpy()
                >= np.ones_like(ms_sp["temperature"].to_numpy()) * -273.15
            ).all(),
        )  # output should be in Celsius
        self.assertTrue((ms_sp["dist"] != ms_sp["dist_in_segment"]).any())

    def test_sample_over_polyline_single_segment_vec_prop(self):
        ms = examples.load_meshseries_HT_2D_XDMF()
        profile = np.array([[4, 2, 0], [4, 18, 0]])
        ms_sp, _ = sample_polyline(
            ms.read(-1),
            "darcy_velocity",
            profile,
        )
        self.assertTrue(not np.any(np.isnan(ms_sp["darcy_velocity_0"])))
        self.assertTrue(
            (
                np.abs(ms_sp["darcy_velocity_0"].to_numpy())
                > np.zeros_like(ms_sp["darcy_velocity_0"].to_numpy())
            ).all()
        )
        self.assertTrue(not np.any(np.isnan(ms_sp["darcy_velocity_1"])))
        self.assertTrue(
            (
                np.abs(ms_sp["darcy_velocity_1"].to_numpy())
                > np.zeros_like(ms_sp["darcy_velocity_1"].to_numpy())
            ).all()
        )
