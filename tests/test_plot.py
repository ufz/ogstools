"""Unit tests for plotting."""

from functools import partial
from tempfile import mkstemp

import matplotlib.pyplot as plt
import numpy as np
import pytest
from pyvista import examples as pv_examples

import ogstools as ogs
from ogstools import examples
from ogstools.plot import utils

assert_allclose = partial(
    np.testing.assert_allclose, rtol=1e-7, atol=1e-100, verbose=True
)


class TestPlotting:
    """Test case for plotting."""

    # TODO: Most of these tests only test for no-throw, currently.

    def test_pyvista_offscreen(self):
        import pyvista as pv

        sphere = pv.Sphere()
        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(sphere)
        plotter.screenshot(filename=None)

    def test_levels(self):
        """Test levels calculation."""
        assert_allclose(
            ogs.plot.compute_levels(0.5, 10.1, 10), [0.5, *range(1, 11), 10.1]
        )
        assert_allclose(
            ogs.plot.compute_levels(293, 350, 10), [293, *range(295, 355, 5)]
        )
        assert_allclose(
            ogs.plot.compute_levels(1e-3, 1.2, 5),
            [1e-3, *np.arange(0.2, 1.4, 0.2)],
        )
        assert_allclose(
            ogs.plot.compute_levels(1e5, 9e6, 20),
            [1e5, *np.arange(5e5, 9.5e6, 5e5)],
        )
        assert_allclose(
            ogs.plot.compute_levels(1, 40, 20), [1, *range(2, 42, 2)]
        )
        assert_allclose(ogs.plot.compute_levels(0.0, 0.0, 10), [0.0, 0.0])
        assert_allclose(ogs.plot.compute_levels(1e9, 1e9, 10), [1e9, 1e9])

    def test_ticklabels(self):
        def compare(lower, upper, precision, ref_labels, ref_offset=None):
            labels, offset = ogs.plot.contourplots.get_ticklabels(
                np.asarray(
                    ogs.plot.compute_levels(lower, upper, n_ticks=precision)
                )
            )
            assert np.all(labels == ref_labels)
            assert offset == ref_offset

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
        labels = utils.justified_labels(points)
        str_lens = np.asarray([len(label) for label in labels])
        assert np.all(str_lens == str_lens[0])

    def test_missing_data(self):
        """Test missing data in mesh."""
        mesh = pv_examples.load_uniform()
        pytest.raises(KeyError, ogs.plot.contourf, mesh, "missing_data")

    def test_plot_2_d(self):
        """Test creation of 2D plots."""
        ogs.plot.setup.reset()
        ogs.plot.setup.material_names = {
            i + 1: f"Layer {i+1}" for i in range(26)
        }
        meshseries = examples.load_meshseries_THM_2D_PVD()
        mesh = meshseries.mesh(1)
        mesh.plot_contourf(ogs.variables.material_id)
        mesh.plot_contourf(ogs.variables.temperature)
        mesh.plot_contourf(ogs.variables.Scalar("pressure_active"))
        ogs.plot.contourf(
            mesh.threshold((1, 3), "MaterialIDs"), ogs.variables.velocity
        )
        fig = mesh.plot_contourf(ogs.variables.displacement[0])
        ogs.plot.shape_on_top(
            fig.axes[0], mesh, lambda x: min(max(0, 0.1 * (x - 3)), 100)
        )
        plt.close()

    def test_diff_plots(self):
        """Test creation of difference plots."""
        meshseries = examples.load_meshseries_THM_2D_PVD()
        mesh0 = meshseries.mesh(0)
        mesh1 = meshseries.mesh(1)
        mesh1.difference(mesh0, "temperature").plot_contourf(
            "temperature_difference"
        )
        for prop in [
            ogs.variables.temperature,
            ogs.variables.displacement,
            ogs.variables.stress,
            ogs.variables.stress.von_Mises,
        ]:
            mesh1.difference(mesh0, prop).plot_contourf(prop)
        plt.close()

    def test_user_defined_ax(self):
        """Test creating plot with subfigures and user provided ax"""
        meshseries = examples.load_meshseries_THM_2D_PVD()
        fig, ax = plt.subplots(3, 1, figsize=(40, 30))
        meshseries.mesh(0).plot_contourf(
            ogs.variables.temperature, fig=fig, ax=ax[0]
        )
        meshseries.mesh(1).plot_contourf(
            ogs.variables.temperature, fig=fig, ax=ax[1]
        )
        diff_mesh = meshseries.mesh(0).difference(
            meshseries.mesh(1), ogs.variables.temperature
        )
        diff_mesh.plot_contourf(ogs.variables.temperature, fig=fig, ax=ax[2])
        plt.close()

    def test_user_defined_ax_two_variables(self):
        """Test creating plot with subfigures and user provided ax with different values plotted"""
        meshseries = examples.load_meshseries_THM_2D_PVD()
        ogs.plot.setup.combined_colorbar = False
        fig, ax = plt.subplots(2, 1, figsize=(40, 20))
        meshseries.mesh(0).plot_contourf(
            ogs.variables.temperature, fig=fig, ax=ax[0]
        )
        meshseries.mesh(1).plot_contourf(
            ogs.variables.displacement, fig=fig, ax=ax[1]
        )
        fig.suptitle("Test user defined ax")
        plt.close()

    def test_user_defined_fig(self):
        """Test creating plot with subfigures and user provided fig"""
        meshseries = examples.load_meshseries_THM_2D_PVD()
        ogs.plot.setup.combined_colorbar = False
        fig, ax = plt.subplots(2, 1, figsize=(40, 20))
        ogs.plot.contourf(
            [meshseries.mesh(0), meshseries.mesh(1)],
            ogs.variables.temperature,
            fig=fig,
        )
        fig.suptitle("Test user defined fig")
        plt.close()

    def test_aggregate_plots(self):
        """Test creation of limit plots."""
        mesh = examples.load_meshseries_CT_2D_XDMF().aggregate_over_time(
            "Si", "var"
        )
        mesh.plot_contourf("Si_var")
        plt.close()

    def test_animation(self):
        """Test creation of animation."""
        meshseries = examples.load_meshseries_THM_2D_PVD()
        timevalues = np.linspace(0, meshseries.timevalues()[-1], num=3)
        anim = meshseries.animate(
            ogs.variables.temperature,
            timevalues,
            mesh_func=lambda mesh: mesh.clip("x"),
            plot_func=lambda ax, t: ax.set_title(str(t)),
        )
        anim.to_jshtml()
        plt.close()

    def test_save_animation(self):
        """Test saving of an animation."""
        meshseries = examples.load_meshseries_THM_2D_PVD()
        timevalues = np.linspace(0, meshseries.timevalues()[-1], num=3)
        anim = meshseries.animate(ogs.variables.temperature, timevalues)
        if not utils.save_animation(anim, mkstemp()[1], 5):
            pytest.skip("Saving animation failed.")
        plt.close()

    def test_plot_3_d(self):
        """Test creation of slice plots for 3D mesh."""
        mesh = pv_examples.load_uniform()
        ogs.plot.contourf(mesh.slice((1, 1, 0)), "Spatial Point Data")
        meshes = np.reshape(mesh.slice_along_axis(4, "x"), (2, 2))
        ogs.plot.contourf(meshes, "Spatial Point Data")
        ogs.plot.contourf(mesh.slice([1, -2, 0]), "Spatial Point Data")
        plt.close()

    def test_streamlines(self):
        """Test streamlines on arbitrarily oriented slices."""
        mesh = pv_examples.load_uniform()
        mesh.point_data["velocity"] = np.random.default_rng().random(
            (mesh.n_points, 3)
        )
        for axis in ["x", "y", "z", [1, 1, 0]]:
            ax: plt.Axes
            fig, ax = plt.subplots()
            i_grid, j_grid, u, v, lw = ogs.plot.vectorplots._vectorfield(
                mesh.slice(axis), ogs.variables.velocity
            )
            for vals in [i_grid, j_grid, u, v, lw]:
                assert not np.all(np.isnan(vals))
            ax.streamplot(
                i_grid, j_grid, u, v, color="k", linewidth=lw, density=1.5
            )
        plt.close()

    def test_xdmf(self):
        """Test creation of 2D plots from xdmf data."""
        mesh = examples.load_meshseries_CT_2D_XDMF().mesh(0)
        mesh.plot_contourf(ogs.variables.saturation)
        plt.close()

    def test_xdmf_with_slices(self):
        """Test creation of 2D plots from xdmf data."""
        mesh = examples.load_meshseries_HT_2D_XDMF().mesh(0)
        mesh.plot_contourf(ogs.variables.pressure)
        plt.close()

    def test_lineplot(self):
        """Test creation of a linesplot from sampled profile data"""
        mesh = examples.load_meshseries_HT_2D_XDMF().mesh(-1)
        profile_HT = np.array([[4, 2, 0], [4, 18, 0]])
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax = mesh.plot_linesample(
            x="dist",
            variable="pressure",
            profile_points=profile_HT,
            ax=ax,
            fontsize=15,
        )
        ax_twinx = ax.twinx()
        ax_twinx = mesh.plot_linesample(
            x="dist",
            variable="temperature",
            profile_points=profile_HT,
            ax=ax_twinx,
            fontsize=15,
        )
        plt.close()

    def test_plot_profile(self):
        """Test creation of a profile plot from sampled profile data"""
        ms_CT = examples.load_meshseries_CT_2D_XDMF()
        profile_CT = np.array([[47.0, 1.17, 72.0], [-4.5, 1.17, -59.0]])
        fig, ax = ms_CT.mesh(11).plot_linesample_contourf(
            ogs.variables.saturation,
            profile_CT,
            resolution=100,
            plot_nodal_pts=True,
        )
        plt.close()
