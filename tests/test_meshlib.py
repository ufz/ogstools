"""Unit tests for meshlib."""

import unittest

import numpy as np

from ogstools.meshlib import MeshSeries, examples


class UtilsTest(unittest.TestCase):
    """Test case for ogstools utilities."""

    def test_all_types(self):
        pvd = MeshSeries(examples.pvd_file)
        vtu = MeshSeries(examples.vtu_file)
        xdmf = MeshSeries(examples.xdmf_file)
        self.assertRaises(TypeError, MeshSeries, __file__)

        for mesh_series in [pvd, xdmf, vtu]:
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

    def test_probe_pvd(self):
        "Test point probing on pvd."
        mesh_series = MeshSeries(examples.pvd_file)
        points = mesh_series.read(0).cell_centers().points
        for method in ["nearest", "probefilter"]:
            values = mesh_series.probe(points, "temperature", method)
            self.assertTrue(not np.any(np.isnan(values)))

    def test_probe_xdmf(self):
        "Test point probing on xdmf."
        mesh_series = MeshSeries(examples.xdmf_file)
        points = mesh_series.read(0).cell_centers().points
        for method in ["nearest", "linear", None]:
            values = mesh_series.probe(points, "temperature", method)
            self.assertTrue(not np.any(np.isnan(values)))
