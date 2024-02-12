"""Unit tests for meshplotlib."""

import unittest
from functools import partial
from tempfile import mkstemp

import numpy as np
from pyvista import examples as pv_examples

from ogstools.meshplotlib import (
    examples,
    plot,
    plot_diff,
    plot_limit,
    plot_probe,
    setup,
)
from ogstools.meshplotlib.animation import animate, save_animation
from ogstools.meshplotlib.levels import get_levels
from ogstools.meshplotlib.plot_features import plot_on_top
from ogstools.meshplotlib.utils import justified_labels
from ogstools.propertylib import Scalar, presets

equality = partial(np.testing.assert_allclose, rtol=1e-7, verbose=True)


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
        equality(get_levels(0.0, 0.0, 10), [0.0, 1e-6])
        equality(get_levels(1e9, 1e9, 10), [1e9, 1e9 + 1e-6])

    def test_justified_labels(self):
        points = np.asarray(
            [
                [x, y, z]
                for x in np.linspace(-1, 0, 3)
                for y in np.linspace(-10, 10, 5)
                for z in np.linspace(1e-6, 1e6, 7)
            ]
        )
        labels = justified_labels(points)
        str_lens = np.asarray([len(label) for label in labels])
        self.assertTrue(np.all(str_lens == str_lens[0]))

    def test_missing_data(self):
        """Test missing data in mesh."""
        mesh = pv_examples.load_uniform()
        self.assertRaises(IndexError, plot, mesh, Scalar("missing_data"))

    def test_plot_2D(self):
        """Test creation of 2D plots."""
        setup.reset()
        setup.length.output_unit = "km"
        setup.material_names = {i + 1: f"Layer {i+1}" for i in range(26)}
        meshseries = examples.meshseries_THM_2D
        mesh = meshseries.read(1)
        plot(mesh, presets.material_id)
        plot(mesh, presets.temperature)
        plot(mesh, Scalar("pressure_active"))
        plot(mesh.threshold((1, 3), "MaterialIDs"), presets.velocity)
        fig = plot(mesh, presets.displacement[0])
        plot_on_top(
            fig.axes[0], mesh, lambda x: min(max(0, 0.1 * (x - 3)), 100)
        )

    def test_diff_plots(self):
        """Test creation of difference plots."""
        meshseries = examples.meshseries_CT_2D
        plot_diff(meshseries.read(0), meshseries.read(1), "Si")

    def test_limit_plots(self):
        """Test creation of limit plots."""
        meshseries = examples.meshseries_CT_2D
        plot_limit(meshseries, "Si", "min")
        plot_limit(meshseries, "Si", "max")

    def test_plot_probe(self):
        """Test creation of probe plots."""
        mesh_series = examples.meshseries_THM_2D
        points = mesh_series.read(0).center
        plot_probe(mesh_series, points, presets.temperature)
        points = mesh_series.read(0).points[[0, -1]]
        plot_probe(mesh_series, points, presets.temperature)
        mesh_series = examples.meshseries_XDMF
        points = mesh_series.read(0).center
        plot_probe(mesh_series, points, presets.temperature)
        mesh_property = presets.velocity.replace(data_name="darcy_velocity")
        plot_probe(mesh_series, points, mesh_property)

    def test_animation(self):
        """Test creation of animation."""
        meshseries = examples.meshseries_THM_2D
        timevalues = np.linspace(0, meshseries.timevalues[-1], num=3)
        titles = [str(tv) for tv in timevalues]
        anim = animate(meshseries, presets.temperature, timevalues, titles)
        anim.to_jshtml()

    def test_save_animation(self):
        """Test saving of an animation."""
        meshseries = examples.meshseries_THM_2D
        timevalues = np.linspace(0, meshseries.timevalues[-1], num=3)
        anim = animate(meshseries, presets.temperature, timevalues)
        if not save_animation(anim, mkstemp()[1], 5):
            self.skipTest("Saving animation failed.")

    def test_plot_3D(self):
        """Test creation of slice plots for 3D mesh."""
        mesh = pv_examples.load_uniform()
        plot(mesh.slice((1, 1, 0)), "Spatial Point Data")
        meshes = np.reshape(mesh.slice_along_axis(4, "x"), (2, 2))
        plot(meshes, "Spatial Point Data")
        plot(mesh.slice([1, -2, 0]), "Spatial Point Data")

    def test_xdmf(self):
        """Test creation of 2D plots from xdmf data."""
        mesh = examples.meshseries_CT_2D.read(0)
        plot(mesh, Scalar("Si"))

    def test_xdmf_with_slices(self):
        """Test creation of 2D plots from xdmf data."""
        mesh = examples.meshseries_XDMF.read(0)
        plot(mesh, presets.pressure)
