"""Unit tests for meshplotlib."""

import unittest
from functools import partial
from tempfile import mkstemp

import matplotlib.pyplot as plt
import numpy as np
from pyvista import examples as pv_examples

from ogstools import examples
from ogstools.meshlib import difference
from ogstools.meshplotlib import (
    clear_labels,
    label_spatial_axes,
    lineplot,
    plot,
    plot_probe,
    plot_profile,
    setup,
    update_font_sizes,
)
from ogstools.meshplotlib.animation import animate, save_animation
from ogstools.meshplotlib.core import get_ticklabels
from ogstools.meshplotlib.levels import compute_levels
from ogstools.meshplotlib.plot_features import _vectorfield, plot_on_top
from ogstools.meshplotlib.utils import justified_labels
from ogstools.propertylib import Scalar, properties

assert_allclose = partial(
    np.testing.assert_allclose, rtol=1e-7, atol=1e-100, verbose=True
)


class MeshplotlibTest(unittest.TestCase):
    """Test case for meshplotlib.

    Most of these tests only test for no-throw, currently."""

    def test_pyvista_offscreen(self):
        import pyvista as pv

        sphere = pv.Sphere()
        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(sphere)
        plotter.screenshot(filename=None)

    def test_levels(self):
        """Test levels calculation property."""
        assert_allclose(
            compute_levels(0.5, 10.1, 10), [0.5, *range(1, 11), 10.1]
        )
        assert_allclose(
            compute_levels(293, 350, 10), [293, *range(295, 355, 5)]
        )
        assert_allclose(
            compute_levels(1e-3, 1.2, 5), [1e-3, *np.arange(0.2, 1.4, 0.2)]
        )
        assert_allclose(
            compute_levels(1e5, 9e6, 20), [1e5, *np.arange(5e5, 9.5e6, 5e5)]
        )
        assert_allclose(compute_levels(1, 40, 20), [1, *range(2, 42, 2)])
        assert_allclose(compute_levels(0.0, 0.0, 10), [0.0, 0.0])
        assert_allclose(compute_levels(1e9, 1e9, 10), [1e9, 1e9])

    def test_ticklabels(self):
        def compare(lower, upper, precision, ref_labels, ref_offset=None):
            labels, offset = get_ticklabels(
                np.asarray(compute_levels(lower, upper, n_ticks=precision))
            )
            self.assertTrue(np.all(labels == ref_labels))
            self.assertEqual(offset, ref_offset)

        compare(1, 10, 6, ["1", "2", "4", "6", "8", "10"])
        compare(1, 10.01, 6, ["1", "2", "4", "6", "8", "10", "10.01"])
        compare(1, 10.001, 6, ["1", "2", "4", "6", "8", "10", "10.001"])
        compare(1, 10.0001, 6, ["1", "2", "4", "6", "8", "10"])
        compare(
            100, 200.1, 6, ["100", "120", "140", "160", "180", "200", "200.1"]
        )
        compare(
            *[-1.2345e-3, 2 + 1.2345e-3, 6],
            ["-0.001", "0", "0.4", "0.8", "1.2", "1.6", "2", "2.001"],
        )
        compare(
            *[-1.2345e-4, 2 + 1.2345e-5, 6],
            ["0", "0.4", "0.8", "1.2", "1.6", "2"],
        )
        compare(
            *[100, 100.0012, 4],
            ["0.0e+00", "4.0e-04", "8.0e-04", "1.2e-03"],
            "100",
        )
        compare(
            *[1.1e5, 1.90001e6, 6],
            ["1.1e+05", "4.0e+05", "8.0e+05", "1.2e+06", "1.6e+06", "1.9e+06"],
        )
        compare(
            *[1e6, 1e6 + 12, 6],
            ["0", "2", "4", "6", "8", "10", "12"],
            "1e+06",
        )

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
        self.assertRaises(KeyError, plot, mesh, Scalar("missing_data"))

    def test_plot_2D(self):
        """Test creation of 2D plots."""
        setup.reset()
        setup.length.output_unit = "km"
        setup.material_names = {i + 1: f"Layer {i+1}" for i in range(26)}
        meshseries = examples.load_meshseries_THM_2D_PVD()
        mesh = meshseries.read(1)
        plot(mesh, properties.material_id)
        plot(mesh, properties.temperature)
        plot(mesh, Scalar("pressure_active"))
        plot(mesh.threshold((1, 3), "MaterialIDs"), properties.velocity)
        fig = plot(mesh, properties.displacement[0])
        plot_on_top(
            fig.axes[0], mesh, lambda x: min(max(0, 0.1 * (x - 3)), 100)
        )
        plt.close()

    def test_diff_plots(self):
        """Test creation of difference plots."""
        meshseries = examples.load_meshseries_THM_2D_PVD()
        mesh0 = meshseries.read(0)
        mesh1 = meshseries.read(1)
        plot(difference(mesh1, mesh0, "temperature"), "temperature_difference")
        for prop in [
            properties.temperature,
            properties.displacement,
            properties.stress,
            properties.stress.von_Mises,
        ]:
            plot(difference(mesh1, mesh0, prop), prop)
        plt.close()

    def test_user_defined_ax(self):
        """Test creating plot with subfigures and user provided ax"""
        meshseries = examples.load_meshseries_THM_2D_PVD()
        fig, ax = plt.subplots(3, 1, figsize=(40, 30))
        plot(meshseries.read(0), properties.temperature, fig=fig, ax=ax[0])
        ax[0].set_title(r"$T(\mathrm{t}_{0})$")
        plot(meshseries.read(1), properties.temperature, fig=fig, ax=ax[1])
        ax[1].set_title(r"$T(\mathrm{t}_{end})$")
        diff_mesh = difference(
            meshseries.read(0), meshseries.read(1), properties.temperature
        )
        plot(diff_mesh, properties.temperature, fig=fig, ax=ax[2])
        ax[2].set_title(r"$T(\mathrm{t}_{end})$-$T(\mathrm{t}_{0})$")
        # fig.suptitle("Test user defined ax")
        fig.tight_layout()
        plt.close()

    def test_user_defined_ax_diff_vals(self):
        """Test creating plot with subfigures and user provided ax with different values plotted"""
        meshseries = examples.load_meshseries_THM_2D_PVD()
        setup.combined_colorbar = False
        fig, ax = plt.subplots(2, 1, figsize=(40, 20))
        plot(meshseries.read(0), properties.temperature, fig=fig, ax=ax[0])
        plot(meshseries.read(1), properties.displacement, fig=fig, ax=ax[1])
        fig.suptitle("Test user defined ax")
        plt.close()

    def test_user_defined_fig(self):
        """Test creating plot with subfigures and user provided fig"""
        meshseries = examples.load_meshseries_THM_2D_PVD()
        setup.combined_colorbar = False
        fig, ax = plt.subplots(2, 1, figsize=(40, 20))
        plot(
            [meshseries.read(0), meshseries.read(1)],
            properties.temperature,
            fig=fig,
        )
        fig.suptitle("Test user defined fig")
        plt.close()

    def test_update_font_sizes(self):
        """Test creating plot with subfigures and user provided fig"""
        meshseries = examples.load_meshseries_THM_2D_PVD()
        setup.combined_colorbar = False
        fig, ax = plt.subplots(2, 1, figsize=(40, 20))
        plot(
            [meshseries.read(0), meshseries.read(1)],
            properties.temperature,
            fig=fig,
        )
        fig, ax = update_font_sizes(fig=fig, fontsize=25)
        fig.suptitle("Test user defined fig")
        plt.close()

    def test_sharexy(self):
        """Test if labels are skipped if axis are shared"""
        meshseries = examples.load_meshseries_THM_2D_PVD()
        mesh_a = meshseries.read(0)
        mesh_b = meshseries.read(1)
        fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
        ax = ax.flatten()
        plot(meshseries.read(0), properties.temperature, fig=fig, ax=ax[0])
        plot(meshseries.read(1), properties.temperature, fig=fig, ax=ax[1])
        diff_ab = difference(mesh_a, mesh_b, properties.temperature)
        diff_ba = difference(mesh_b, mesh_a, properties.temperature)
        plot(diff_ab, properties.temperature, fig=fig, ax=ax[2])
        plot(diff_ba, properties.temperature, fig=fig, ax=ax[3])
        plt.close()

    def test_label_sharedxy(self):
        """Test labeling shared x and y axes"""
        meshseries = examples.load_meshseries_THM_2D_PVD()
        mesh_a = meshseries.read(0)
        mesh_b = meshseries.read(1)
        fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
        plot(meshseries.read(0), properties.temperature, fig=fig, ax=ax[0][0])
        plot(meshseries.read(1), properties.temperature, fig=fig, ax=ax[1][0])
        diff_ab = difference(mesh_a, mesh_b, properties.temperature)
        diff_ba = difference(mesh_b, mesh_a, properties.temperature)
        plot(diff_ab, properties.temperature, fig=fig, ax=ax[0][1])
        plot(diff_ba, properties.temperature, fig=fig, ax=ax[1][1])
        label_spatial_axes(ax, "x", "y")
        plt.close()

    def test_spatial_label(self):
        """Test axes labeling"""
        fig, ax = plt.subplots(2, 2)
        label_spatial_axes(ax, "x", "y")
        plt.close()

    def test_spatial_label_clear(self):
        """Test axes labels clearing"""
        fig, ax = plt.subplots(2, 2)
        label_spatial_axes(ax, "x", "y")
        clear_labels(ax)
        plt.close()

    def test_limit_plots(self):
        """Test creation of limit plots."""
        mesh = examples.load_meshseries_CT_2D_XDMF().aggregate("Si", "var")
        plot(mesh, "Si_var")
        plt.close()

    def test_plot_probe(self):
        """Test creation of probe plots."""
        meshseries = examples.load_meshseries_THM_2D_PVD()
        points = meshseries.read(0).center
        plot_probe(meshseries, points, properties.temperature)
        points = meshseries.read(0).points[[0, -1]]
        plot_probe(meshseries, points, properties.temperature)
        plot_probe(meshseries, points, properties.velocity)
        plot_probe(meshseries, points, properties.stress)
        plot_probe(meshseries, points, properties.stress.von_Mises)
        mesh_series = examples.load_meshseries_HT_2D_XDMF()
        points = mesh_series.read(0).center
        plot_probe(mesh_series, points, properties.temperature)
        # TODO: fix darcy_velocity vs velocity problem @ FZ & FK
        mesh_property = properties.velocity.replace(data_name="darcy_velocity")
        plot_probe(mesh_series, points, mesh_property)
        plt.close()

    def test_animation(self):
        """Test creation of animation."""
        meshseries = examples.load_meshseries_THM_2D_PVD()
        timevalues = np.linspace(0, meshseries.timevalues[-1], num=3)
        titles = [str(tv) for tv in timevalues]
        anim = animate(meshseries, properties.temperature, timevalues, titles)
        anim.to_jshtml()
        plt.close()

    def test_save_animation(self):
        """Test saving of an animation."""
        meshseries = examples.load_meshseries_THM_2D_PVD()
        timevalues = np.linspace(0, meshseries.timevalues[-1], num=3)
        anim = animate(meshseries, properties.temperature, timevalues)
        if not save_animation(anim, mkstemp()[1], 5):
            self.skipTest("Saving animation failed.")
        plt.close()

    def test_plot_3D(self):
        """Test creation of slice plots for 3D mesh."""
        mesh = pv_examples.load_uniform()
        plot(mesh.slice((1, 1, 0)), "Spatial Point Data")
        meshes = np.reshape(mesh.slice_along_axis(4, "x"), (2, 2))
        plot(meshes, "Spatial Point Data")
        plot(mesh.slice([1, -2, 0]), "Spatial Point Data")
        plt.close()

    def test_streamlines(self):
        """Test streamlines on arbitrarily oriented slices."""
        mesh = pv_examples.load_uniform()
        mesh.point_data["velocity"] = np.random.default_rng().random(
            (mesh.n_points, 3)
        )
        for axis in ["x", "y", "z", [1, 1, 0]]:
            fig, ax = plt.subplots()
            i_grid, j_grid, u, v, lw = _vectorfield(
                mesh.slice(axis), properties.velocity
            )
            for vals in [i_grid, j_grid, u, v, lw]:
                assert not np.all(np.isnan(vals))
            ax.streamplot(
                i_grid, j_grid, u, v, color="k", linewidth=lw, density=1.5
            )
        plt.close()

    def test_xdmf(self):
        """Test creation of 2D plots from xdmf data."""
        mesh = examples.load_meshseries_CT_2D_XDMF().read(0)
        plot(mesh, Scalar("Si"))
        plt.close()

    def test_xdmf_with_slices(self):
        """Test creation of 2D plots from xdmf data."""
        mesh = examples.load_meshseries_HT_2D_XDMF().read(0)
        plot(mesh, properties.pressure)
        plt.close()

    def test_lineplot(self):
        """Test creation of a linesplot from sampled profile data"""
        ms_HT = examples.load_meshseries_HT_2D_XDMF()
        profile_HT = np.array([[4, 2, 0], [4, 18, 0]])
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax = lineplot(
            x="dist",
            y=["pressure", "temperature"],
            mesh=ms_HT.read(-1),
            profile_points=profile_HT,
            ax=ax,
            twinx=True,
            fontsize=15,
        )
        plt.close()

    def test_plot_profile(self):
        """Test creation of a profile plot from sampled profile data"""
        ms_CT = examples.load_meshseries_CT_2D_XDMF()
        profile_CT = np.array(
            [
                [47.0, 1.17, 72.0],  # Point A
                [-4.5, 1.17, -59.0],  # Point B
            ]
        )
        si = Scalar(
            data_name="Si",
            data_unit="",
            output_unit="%",
            output_name="Saturation",
        )
        fig, ax = plot_profile(
            ms_CT.read(11),
            si,
            profile_CT,
            resolution=100,
            plot_nodal_pts=True,
            profile_plane=[0, 2],  # This profile is in XZ plane, not XY!
        )
        fig, ax = update_font_sizes(label_axes="none", fig=fig)
        fig.tight_layout()
