"""Unit tests for meshplotlib."""

import unittest
from functools import partial
from pathlib import Path

import numpy as np
from pyvista import examples as pv_examples

from ogstools.meshplotlib import MeshSeries, examples, plot, setup
from ogstools.meshplotlib.levels import get_levels
from ogstools.propertylib import THM, ScalarProperty

THIS_DIR = Path(__file__).parent

equality = partial(np.testing.assert_allclose, rtol=1e-7, verbose=True)
setup.show_fig_after_plot = False


class MeshplotlibTest(unittest.TestCase):
    """Test case for meshplotlib."""

    def test_pyvista_offscreen(self):
        import pyvista as pv

        sphere = pv.Sphere()
        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(sphere)
        plotter.screenshot(filename=None)

    def test_levels(self):
        """Test levels calculation property."""
        equality(get_levels(0.5, 10.1, 10), [0.5, *range(1, 11), 10.1])
        equality(get_levels(293, 350, 10), [293, *range(295, 355, 5)])
        equality(get_levels(1e-3, 1.2, 5), [1e-3, *np.arange(0.2, 1.4, 0.2)])
        equality(get_levels(1e5, 9e6, 20), [1e5, *np.arange(5e5, 9.5e6, 5e5)])
        equality(get_levels(1, 40, 20), [1, *range(2, 42, 2)])

    def test_missing_data(self):
        """Test missing data in mesh."""
        mesh = pv_examples.load_uniform()
        self.assertRaises(
            IndexError, plot, mesh, ScalarProperty("missing_data")
        )

    def test_plot_2D(self):
        """Test creation of 2D plots."""
        setup.reset()
        setup.length.output_unit = "km"
        setup.material_names = {i + 1: f"Layer {i+1}" for i in range(26)}
        meshseries = examples.meshseries_THM_2D
        plot(meshseries.read(0), property=THM.material_id)
        plot(meshseries.read(1), property=THM.temperature)
        plot(meshseries.read(1), ScalarProperty("pressure_active"))
        plot(meshseries.read(1).threshold((1, 3), "MaterialIDs"), THM.velocity)
        plot(meshseries.read(1), THM.displacement[0])

    def test_plot_3D(self):
        """Test creation of slice plots for 3D mesh."""
        mesh = pv_examples.load_uniform()
        # TODO: find alternative for isometric plot with pyvista
        plot(mesh, ScalarProperty("Spatial Point Data"))
        plot(mesh.slice((1, 1, 0)), ScalarProperty("Spatial Point Data"))
        meshes = np.reshape(mesh.slice_along_axis(4, "x"), (2, 2))
        plot(meshes, ScalarProperty("Spatial Point Data"))
        plot(mesh.slice([1, -2, 0]), ScalarProperty("Spatial Point Data"))

    def test_xdmf(self):
        """Test creation of 2D plots from xdmf data."""
        filename = pv_examples.download_meshio_xdmf(load=False)
        mesh = MeshSeries(filename).read(0)
        plot(mesh, ScalarProperty("phi"))
        # plot(mesh, VectorProperty("u")) # TODO: property not in data?
        plot(mesh, ScalarProperty("a"))

    def test_xdmf_with_slices(self):
        """Test creation of 2D plots from xdmf data."""
        mesh = MeshSeries(
            f"{THIS_DIR}/data/meshplotlib/2D_single_fracture_HT_2D_single_fracture.xdmf"
        ).read(0)
        plot(mesh, property=THM.temperature)


if __name__ == "__main__":
    unittest.main(argv=[""], verbosity=2, exit=False)