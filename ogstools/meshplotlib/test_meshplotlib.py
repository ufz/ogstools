"""Unit tests for meshplotlib."""

import sys
import unittest
from functools import partial
from pathlib import Path

import numpy as np
from pyvista import examples

from ogstools.meshplotlib import MeshSeries, plot, setup
from ogstools.meshplotlib.levels import get_levels
from ogstools.propertylib import THM, ScalarProperty, VectorProperty

THIS_DIR = Path(__file__).parent

equality = partial(np.testing.assert_allclose, rtol=1e-7, verbose=True)
setup.show_fig_after_plot = False


class MeshplotlibTest(unittest.TestCase):
    """Test case for meshplotlib."""

    def test_levels(self):
        """Test levels calculation property."""
        equality(get_levels(0.5, 10.1, 10), [0.5, *range(1, 11), 10.1])
        equality(get_levels(293, 350, 10), [293, *range(295, 355, 5)])
        equality(get_levels(1e-3, 1.2, 5), [1e-3, *np.arange(0.2, 1.4, 0.2)])
        equality(get_levels(1e5, 9e6, 20), [1e5, *np.arange(5e5, 9.5e6, 5e5)])
        equality(get_levels(1, 40, 20), [1, *range(2, 42, 2)])

    def test_missing_data(self):
        """Test missing data in mesh."""
        mesh = examples.load_uniform()
        fig = plot(mesh, ScalarProperty("missing_data"))
        assert fig is None, "Figure should be empty."

    def test_plot_2D(self):
        """Test creation of 2D plots."""
        setup.reset()
        setup.length.output_unit = "km"
        setup.material_names = {i + 1: f"Layer {i+1}" for i in range(26)}
        meshseries = MeshSeries(f"{THIS_DIR}/examples/2D.pvd")
        plot(meshseries.read(0), property=THM.material_id)
        plot(meshseries.read(1), property=THM.temperature)
        plot(meshseries.read(1), ScalarProperty("pressure_active"))
        plot(meshseries.read(1).threshold((1, 3), "MaterialIDs"), THM.velocity)
        plot(meshseries.read(1), THM.displacement.component(0))

    def test_plot_3D(self):
        """Test creation of slice plots for 3D mesh."""
        mesh = examples.load_uniform()
        # TODO: find alternative for isometric plot with pyvista
        if "win" not in sys.platform:
            # pyvista.start_xvfb() doesn't work on windows
            plot(mesh, ScalarProperty("Spatial Point Data"))
        plot(mesh.slice((1, 1, 0)), ScalarProperty("Spatial Point Data"))
        meshes = np.reshape(mesh.slice_along_axis(4, "x"), (2, 2))
        plot(meshes, ScalarProperty("Spatial Point Data"))
        plot(mesh.slice([1, -2, 0]), ScalarProperty("facies"))

    def test_xdmf(self):
        """Test creation of 2D plots from xdmf data."""
        filename = examples.download_meshio_xdmf(load=False)
        mesh = MeshSeries(filename).read(0)
        plot(mesh, ScalarProperty("phi"))
        plot(mesh, VectorProperty("u"))
        plot(mesh, ScalarProperty("a"))


if __name__ == "__main__":
    unittest.main(argv=[""], verbosity=2, exit=False)
