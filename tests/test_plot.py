"""Unit tests for plotting."""

from functools import partial
from tempfile import mkstemp

import matplotlib.pyplot as plt
import numpy as np
import pytest
from pyvista import examples as pv_examples

import ogstools as ot
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
            ot.plot.compute_levels(0.5, 10.1, 10), [0.5, *range(1, 11), 10.1]
        )
        assert_allclose(
            ot.plot.compute_levels(293, 350, 10), [293, *range(295, 355, 5)]
        )
        assert_allclose(
            ot.plot.compute_levels(1e-3, 1.2, 5),
            [1e-3, *np.arange(0.2, 1.4, 0.2)],
        )
        assert_allclose(
            ot.plot.compute_levels(1e5, 9e6, 20),
            [1e5, *np.arange(5e5, 9.5e6, 5e5)],
        )
        assert_allclose(
            ot.plot.compute_levels(1, 40, 20), [1, *range(2, 42, 2)]
        )
        assert_allclose(ot.plot.compute_levels(0.0, 0.0, 10), [0.0, 0.0])
        assert_allclose(ot.plot.compute_levels(1e9, 1e9, 10), [1e9, 1e9])

    def test_ticklabels(self):
        def compare(lower, upper, precision, ref_labels, ref_offset=None):
            labels, offset = ot.plot.contourplots.get_ticklabels(
                np.asarray(
                    ot.plot.compute_levels(lower, upper, n_ticks=precision)
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
        pytest.raises(KeyError, ot.plot.contourf, mesh, "missing_data")

    def test_plot_2_d(self):
        """Test creation of 2D plots."""
        ot.plot.setup.reset()
        ot.plot.setup.material_names = {
            i + 1: f"Layer {i+1}" for i in range(26)
        }
        meshseries = examples.load_meshseries_THM_2D_PVD()
        mesh = meshseries.mesh(1)
        mesh.plot_contourf(ot.variables.material_id)
        mesh.plot_contourf(ot.variables.temperature)
        mesh.plot_contourf(ot.variables.Scalar("pressure_active"))
        ot.plot.contourf(
            mesh.threshold((1, 3), "MaterialIDs"), ot.variables.velocity
        )
        fig = mesh.plot_contourf(ot.variables.displacement[0])
        ot.plot.shape_on_top(
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
            ot.variables.temperature,
            ot.variables.displacement,
            ot.variables.stress,
            ot.variables.stress.von_Mises,
        ]:
            mesh1.difference(mesh0, prop).plot_contourf(prop)
        plt.close()

    def test_user_defined_ax(self):
        """Test creating plot with subfigures and user provided ax"""
        meshseries = examples.load_meshseries_THM_2D_PVD()
        fig, ax = plt.subplots(3, 1, figsize=(40, 30))
        meshseries.mesh(0).plot_contourf(
            ot.variables.temperature, fig=fig, ax=ax[0]
        )
        meshseries.mesh(1).plot_contourf(
            ot.variables.temperature, fig=fig, ax=ax[1]
        )
        diff_mesh = meshseries.mesh(0).difference(
            meshseries.mesh(1), ot.variables.temperature
        )
        diff_mesh.plot_contourf(ot.variables.temperature, fig=fig, ax=ax[2])
        plt.close()

    def test_user_defined_ax_two_variables(self):
        """Test creating plot with subfigures and user provided ax with different values plotted"""
        meshseries = examples.load_meshseries_THM_2D_PVD()
        ot.plot.setup.combined_colorbar = False
        fig, ax = plt.subplots(2, 1, figsize=(40, 20))
        meshseries.mesh(0).plot_contourf(
            ot.variables.temperature, fig=fig, ax=ax[0]
        )
        meshseries.mesh(1).plot_contourf(
            ot.variables.displacement, fig=fig, ax=ax[1]
        )
        fig.suptitle("Test user defined ax")
        plt.close()

    def test_user_defined_fig(self):
        """Test creating plot with subfigures and user provided fig"""
        meshseries = examples.load_meshseries_THM_2D_PVD()
        ot.plot.setup.combined_colorbar = False
        fig, ax = plt.subplots(2, 1, figsize=(40, 20))
        ot.plot.contourf(
            [meshseries.mesh(0), meshseries.mesh(1)],
            ot.variables.temperature,
            fig=fig,
        )
        fig.suptitle("Test user defined fig")
        plt.close()

    def test_aggregate_plots(self):
        """Test creation of limit plots."""
        mesh = examples.load_meshseries_CT_2D_XDMF().aggregate_over_time(
            "Si", np.var
        )
        mesh.plot_contourf("Si_var")
        plt.close()

    def test_animation(self):
        """Test creation of animation."""
        meshseries = examples.load_meshseries_THM_2D_PVD()
        timevalues = np.linspace(0, meshseries.timevalues[-1], num=3)
        anim = meshseries.animate(
            ot.variables.temperature,
            timevalues,
            mesh_func=lambda mesh: mesh.clip("x"),
            plot_func=lambda ax, t: ax.set_title(str(t)),
        )
        anim.to_jshtml()
        plt.close()

    def test_save_animation(self):
        """Test saving of an animation."""
        meshseries = examples.load_meshseries_THM_2D_PVD()
        timevalues = np.linspace(0, meshseries.timevalues[-1], num=3)
        anim = meshseries.animate(ot.variables.temperature, timevalues)
        if not utils.save_animation(anim, mkstemp()[1], 5):
            pytest.skip("Saving animation failed.")
        plt.close()

    def test_plot_3_d(self):
        """Test creation of slice plots for 3D mesh."""
        mesh = pv_examples.load_uniform()
        ot.plot.contourf(mesh.slice((1, 1, 0)), "Spatial Point Data")
        meshes = np.reshape(mesh.slice_along_axis(4, "x"), (2, 2))
        ot.plot.contourf(meshes, "Spatial Point Data")
        ot.plot.contourf(mesh.slice([1, -2, 0]), "Spatial Point Data")
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
            i_grid, j_grid, u, v, lw = ot.plot.vectorplots._vectorfield(
                mesh.slice(axis), ot.variables.velocity
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
        mesh.plot_contourf(ot.variables.saturation)
        plt.close()

    def test_xdmf_with_slices(self):
        """Test creation of 2D plots from xdmf data."""
        mesh = examples.load_meshseries_HT_2D_XDMF().mesh(0)
        mesh.plot_contourf(ot.variables.pressure)
        plt.close()

    def test_lineplot(self):
        """Test creation of a linesplot from sampled profile data"""
        ot.plot.setup.set_units(spatial="km", time="a")
        mesh = examples.load_meshseries_THM_2D_PVD().mesh(-1)
        mesh.points[:, 2] = 0.0
        x1, x2, y1, y2 = mesh.bounds[:4]
        xc, yc, z = mesh.center
        sample_x = mesh.sample_over_line([x1, yc, z], [x2, yc, z])
        sample_y = mesh.sample_over_line([xc, y1, z], [xc, y2, z])
        sample_xy = mesh.sample_over_line([x1, y1, z], [x2, y2, z])
        sample_xz = mesh.rotate_x(90).sample_over_line([x1, 0, y1], [x2, 0, y2])
        sample_yz = mesh.rotate_y(90).sample_over_line([0, y1, x1], [0, y2, x2])

        def check(*args, x_l: str, y_l: str) -> None:
            fig = ot.plot.line(*args, figsize=[4, 3])
            assert fig.axes[0].get_xlabel().split(" ")[0] == x_l
            assert fig.axes[0].get_ylabel().split(" ")[0] == y_l

        check(sample_x, ot.variables.temperature, x_l="x", y_l="temperature")
        check(sample_x, x_l="x", y_l="y")
        check(sample_y, ot.variables.temperature, x_l="y", y_l="temperature")
        check(sample_y, x_l="x", y_l="y")
        check(sample_xy, ot.variables.temperature, x_l="x", y_l="temperature")
        check(sample_xy, x_l="x", y_l="y")
        check(sample_xz, ot.variables.temperature, x_l="x", y_l="temperature")
        check(sample_xz, x_l="x", y_l="z")
        check(sample_yz, ot.variables.temperature, x_l="y", y_l="temperature")
        check(sample_yz, x_l="y", y_l="z")
        check(sample_yz, "z", "y", x_l="z", y_l="y")
        check(sample_x, "x", "temperature", x_l="x", y_l="temperature")
        check(sample_y, "temperature", "y", x_l="temperature", y_l="y")
        check(sample_xy, ot.variables.displacement, ot.variables.temperature,
              x_l="displacement", y_l="temperature")  # fmt: skip
        _, ax = plt.subplots(figsize=[4, 3])
        ot.plot.line(sample_y, ot.variables.pressure, "x", ax=ax)
        _ = ot.plot.line(
            sample_y, "y", "x", figsize=[5, 5], color="g", linewidth=1,
            ls="--", label="test", grid=True,
        )  # fmt: skip
        with pytest.raises(TypeError):
            ot.plot.line(sample_y, ax)
