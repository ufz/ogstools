"""Unit tests for plotting."""

from collections.abc import Callable
from pathlib import Path
from tempfile import mkstemp
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
import pytest
import pyvista as pv
from matplotlib.animation import FFMpegWriter
from matplotlib.animation import ImageMagickWriter as IMWriter
from pyvista import examples as pv_examples

import ogstools as ot
from ogstools import examples
from ogstools.plot import contourf, utils


class TestPlotting:
    """Test case for plotting."""

    # ### Testing Utils ########################################################

    @pytest.mark.parametrize(
        ("desired", "lower", "upper", "n_ticks"),
        [
            ([0.5, *range(1, 11), 10.1], 0.5, 10.1, 10),
            ([293, *range(295, 355, 5)], 293, 350, 10),
            ([1e-3, *np.arange(0.2, 1.4, 0.2)], 1e-3, 1.2, 5),
            ([1e5, *np.arange(5e5, 9.5e6, 5e5)], 1e5, 9e6, 20),
            ([1, *range(2, 42, 2)], 1, 40, 20),
            ([0.0, 0.0], 0.0, 0.0, 10),
            ([1e9, 1e9], 1e9, 1e9, 10),
        ],
    )
    def test_levels(
        self, desired: list, lower: float, upper: float, n_ticks: int
    ):
        """Test calculation of nicely spaced levels."""
        actual = ot.plot.compute_levels(lower, upper, n_ticks)
        np.testing.assert_allclose(
            actual, desired, rtol=1e-7, atol=1e-100, verbose=True
        )

    @pytest.mark.parametrize(
        ("lower", "upper", "n_ticks", "ref_labels", "ref_offset"),
        [
            (1, 10, 6, ["1", "2", "4", "6", "8", "10"], None),
            (1, 10.01, 6, ["1", "2", "4", "6", "8", "10", "10.01"], None),
            (1, 10.001, 6, ["1", "2", "4", "6", "8", "10", "10.001"], None),
            (1, 10.0001, 6, ["1", "2", "4", "6", "8", "10"], None),
            (100, 200.1, 6, ["100", "120", "140", "160", "180", "200", "200.1"], None),
            (*[-1.2345e-3, 2 + 1.2345e-3, 6], ["-0.001", "0", "0.4", "0.8", "1.2", "1.6", "2", "2.001"], None),
            (*[-1.2345e-4, 2 + 1.2345e-5, 6], ["0", "0.4", "0.8", "1.2", "1.6", "2"], None),
            (*[1.1e5, 1.90001e6, 6], ["1.1e+05", "4.0e+05", "8.0e+05", "1.2e+06", "1.6e+06", "1.9e+06"], None),
            (*[100, 100.0012, 4], ["0.0e+00", "4.0e-04", "8.0e-04", "1.2e-03"], "100"),
            (*[1e6, 1e6 + 12, 6], ["0", "2", "4", "6", "8", "10", "12"], "1e+06"),
        ],
    )  # fmt: skip
    def test_ticklabels(
        self, lower: float, upper: float, n_ticks: int,
        ref_labels: list, ref_offset: str | None,
    ):  # fmt: skip
        """Check for equality of ticklabels and expected labels."""
        levels = ot.plot.compute_levels(lower, upper, n_ticks=n_ticks)
        labels, offset = ot.plot.contourplots.get_ticklabels(levels)
        assert np.all(labels == ref_labels)
        assert offset == ref_offset

    def test_justified_labels(self):
        """test that justified labels have the same length."""
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

    def test_colors_from_cmap(self):
        """Test that colors created from a colormap have the correct length."""
        for cmap_color in ["tab10", "Blues", ["r", "g", "b"]]:
            for n in [2, 5, 10]:
                colors = ot.plot.utils.colors_from_cmap(cmap_color, n)
                assert len(colors) == n

    # ### Testing contourf #####################################################

    def test_missing_data(self):
        """Test missing data in mesh."""
        mesh = pv_examples.load_uniform()
        pytest.raises(KeyError, ot.plot.contourf, mesh, "missing_data")

    var_params: ClassVar[list[tuple[ot.variables.Variable, dict]]] = [
        (ot.variables.material_id, {}),
        (ot.variables.temperature, {"continuous_cmap": True}),
        (ot.variables.pressure.get_mask(), {}),
        (ot.variables.velocity, {"arrowsize": 2}),
        (ot.variables.displacement[1], {"log_scaled": True}),
        (ot.variables.stress, {}),
        (ot.variables.stress.von_Mises, {}),
    ]

    @pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 20})
    @pytest.mark.parametrize(("var", "kwargs"), var_params)
    def test_contourf(
        self, var: ot.variables.Variable, kwargs: dict
    ) -> plt.Figure:
        """Test filled 2D contourplots via image comparison."""
        ot.plot.setup.reset()
        ot.plot.setup.material_names = {
            i + 1: f"Layer {i+1}" for i in range(26)
        }
        ms = examples.load_meshseries_THM_2D_PVD()
        return contourf(ms[1], var, **kwargs)

    @pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 20})
    def test_shape_on_top(self) -> plt.Figure:
        "Test plotting the shape on top of a contourplot via image comparison."
        ms = examples.load_meshseries_THM_2D_PVD()
        fig = contourf(ms[-1], ot.variables.displacement)
        ot.plot.shape_on_top(
            fig.axes[0], ms[-1], lambda x: min(max(0, 0.1 * (x - 3)), 100)
        )
        return fig

    @pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 20})
    @pytest.mark.parametrize(("var", "kwargs"), var_params)
    def test_contourf_diff(
        self, var: ot.variables.Variable, kwargs: dict
    ) -> plt.Figure:
        """Test creation of difference plots via image comparison."""
        ms = examples.load_meshseries_THM_2D_PVD()
        return contourf(
            ot.meshlib.difference(ms[1], ms[0], var), var.difference, **kwargs
        )

    @pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 20})
    def test_user_defined_ax(self) -> plt.Figure:
        """Test plotting with provided ax via image comparison."""
        ms = examples.load_meshseries_THM_2D_PVD()
        diff = ot.meshlib.difference(ms[1], ms[0], ot.variables.temperature)
        fig, ax = plt.subplots(3, 1, figsize=(40, 30))
        contourf(ms[0], ot.variables.temperature, fig=fig, ax=ax[0])
        contourf(ms[1], ot.variables.temperature, fig=fig, ax=ax[1])
        contourf(diff, ot.variables.temperature.difference, fig=fig, ax=ax[2])
        return fig

    @pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 20})
    def test_user_defined_ax_different_vars(self) -> plt.Figure:
        """Test plotting with different vars in one fig via image comparison."""
        ot.plot.setup.combined_colorbar = False
        ms = examples.load_meshseries_THM_2D_PVD()
        fig, ax = plt.subplots(2, 1, figsize=(40, 20))
        contourf(ms[0], ot.variables.temperature, fig=fig, ax=ax[0])
        contourf(ms[1], ot.variables.displacement, fig=fig, ax=ax[1])
        return fig

    @pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 20})
    def test_user_defined_fig(self) -> plt.Figure:
        """Test plotting with provided fig but not ax via image comparison."""
        ot.plot.setup.combined_colorbar = False
        ms = examples.load_meshseries_THM_2D_PVD()
        fig, ax = plt.subplots(2, 1, figsize=(40, 20))
        contourf([ms[0], ms[1]], ot.variables.temperature, fig=fig)
        return fig

    @pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 20})
    def test_aggregate_over_time(self) -> plt.Figure:
        """Test creation of temporal aggregation plots via image comparison."""
        ms = examples.load_meshseries_CT_2D_XDMF()
        mesh = ms.aggregate_temporal("Si", np.var)
        return contourf(mesh, "Si_var")

    @pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 50})
    def test_domain_aggregate(self):
        """Test creation of spatial aggregation plots via image comparison."""
        mesh_series = examples.load_meshseries_THM_2D_PVD()
        return mesh_series.plot_spatial_aggregate("temperature", np.mean)

    @pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 20})
    @pytest.mark.parametrize(
        ("args", "kwargs"),
        [
            (("x", "time"), {"cb_loc": "left", "levels": range(0, 110, 10)}),
            (("x", "time"), {"log_scaled": True}),
            (("time", "x"), {"time_logscale": True}),
        ],
    )
    def test_time_slice(self, args, kwargs):
        """Test creation of time slice plots via image comparison."""
        results = examples.load_meshseries_CT_2D_XDMF()
        points = np.linspace([0, 0, 70], [150, 0, 70], num=100)
        ms_pts = results.probe(points)
        return ms_pts.plot_time_slice(*args, variable="Saturation", **kwargs)

    def test_time_slice_failing_args(self):
        """Test time slice for expected errors when given wrong arguments."""
        results = examples.load_meshseries_HT_2D_XDMF()
        points = np.linspace([2, 2, 0], [4, 18, 0], num=100)
        ms_pts = results.probe(points, "temperature")
        fig, _ = plt.subplots()
        with pytest.raises(ValueError, match="fig and ax together"):
            _ = ms_pts.plot_time_slice("x", "time", "temperature", fig=fig)
        with pytest.raises(KeyError, match="has to be 'time'"):
            _ = ms_pts.plot_time_slice("x", "y", "temperature")
        with pytest.raises(KeyError, match="has to be a spatial"):
            _ = ms_pts.plot_time_slice("time", "temperature", "y")

    @pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 20})
    @pytest.mark.parametrize(
        ("slice_func"),
        [
            lambda mesh: np.reshape(mesh.slice_along_axis(4, "x"), (2, 2)),
            lambda mesh: mesh.slice((1, 1, 0)),
            lambda mesh: mesh.slice([1, -2, 0]),
        ],
    )
    def test_contourf_arbitrary_orientation(self, slice_func):
        "Test with arbitrary orientated 2D mesh via image comparison."
        mesh = pv_examples.load_uniform()
        return contourf(slice_func(mesh), "Spatial Point Data")

    @pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 30})
    @pytest.mark.parametrize(
        ("axis"), ["x", "y", "z", [1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]]
    )
    def test_streamlines(self, axis) -> plt.Figure:
        "Test streamlines on arbitrarily oriented slices via image comparison."
        mesh = pv_examples.load_uniform()
        mesh.point_data["velocity"] = mesh.points
        fig, ax = plt.subplots()
        ot.plot.vectorplots.streamlines(
            mesh.slice(axis), ax=ax, variable=ot.variables.velocity
        )
        ax.margins(0)
        fig.tight_layout()
        return fig

    @pytest.mark.parametrize(
        ("mesh", "var"),
        [(examples.load_meshseries_CT_2D_XDMF()[-1], ot.variables.saturation),
         (examples.load_meshseries_HT_2D_XDMF()[-1], ot.variables.pressure)],
    )  # fmt: skip
    @pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 20})
    def test_xdmf(self, mesh, var) -> plt.Figure:
        """Test creation of 2D plots from xdmf data via image comparison."""
        return contourf(mesh, var)

    @pytest.mark.xfail(reason="Image files did not match.")
    @pytest.mark.mpl_image_compare(
        savefig_kwargs={"dpi": 20}, filename="test_xdmf.png"
    )
    def test_fail_image_compare(self, config) -> plt.Figure:
        "Test comparison of images actually fails upon differences."
        if config.getoption("--mpl-generate-path"):
            pytest.skip("Skipping to not override the reference figure.")
        mesh = examples.load_meshseries_CT_2D_XDMF().mesh(0)
        mesh["Si"] += 1
        return contourf(mesh, ot.variables.saturation)

    # ### Testing plot line ####################################################

    @pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 30})
    @pytest.mark.parametrize("var", [ot.variables.temperature, None])
    @pytest.mark.parametrize(
        ("name", "ids", "view", "func"),
        [
            ("x", [0], slice(0, 2), lambda m: m),
            ("y", [1], slice(2, 4), lambda m: m),
            ("xy", [0, 1], slice(0, 4), lambda m: m),
            ("xz", [0, 2, 1], slice(None), lambda m: m.rotate_x(90)),
            ("yz", [2, 1, 0], slice(None), lambda m: m.rotate_y(-90)),
        ],
    )
    def test_line_linesample(
        self,
        var: ot.variables.Variable | None,
        name: str,
        view: slice,
        ids: int | list,
        func: Callable[[pv.UnstructuredGrid], pv.UnstructuredGrid],
    ) -> plt.Figure:
        """Test plot.line from sampled profile data via image comparison."""
        mesh = examples.load_meshseries_THM_2D_PVD().mesh(-1)
        mesh.points[:, 2] = 0
        obs_pts = np.reshape(mesh.center * 2, (2, 3))
        obs_pts[:, ids] = np.reshape(mesh.bounds[view], (len(ids), 2)).T
        sample = func(mesh).sample_over_line(obs_pts[0], obs_pts[1])
        if var is not None:
            sample[var.data_name][sample[var.data_name] == 0.0] = np.nan
        fig = ot.plot.line(sample, var1=var, label=f"{name}-sampling line")
        fig.tight_layout()
        return fig

    @pytest.mark.parametrize(
        ("var1", "var2"),
        [("time", ot.variables.saturation), (ot.variables.saturation, "time")],
    )
    @pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 30})
    def test_line_meshseries(
        self, var1: ot.variables.Variable, var2: ot.variables.Variable
    ) -> plt.Figure:
        """Test plot.line from sampled meshseries data via image comparison."""
        ms = examples.load_meshseries_CT_2D_XDMF()
        pts = np.linspace([0, 0, 60], [120, 0, 60], 4)
        ms_pts = ms.probe(pts)
        labels = [f"{i}: x={pt[0]: >5} z={pt[2]}" for i, pt in enumerate(pts)]
        fig = ot.plot.line(
            ms_pts, var1, var2, labels=labels, monospace=True, outer_legend=True
        )
        fig.tight_layout()
        return fig

    @pytest.mark.parametrize(
        ("var1", "var2"),
        [
            (ot.variables.displacement["y"], ot.variables.temperature),
            (ot.variables.effective_pressure, ot.variables.stress.von_Mises),
        ],
    )
    @pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 30})
    def test_spatial_phaseplot(
        self, var1: ot.variables.Variable, var2: ot.variables.Variable
    ) -> plt.Figure:
        """Test plot.line over 2 vars spatially via image comparison."""
        mesh = examples.load_meshseries_THM_2D_PVD()[-1]
        obs_pts = np.reshape(mesh.center * 2, (2, 3))
        obs_pts[:, 1] = mesh.bounds[2:4]
        sample = mesh.sample_over_line(obs_pts[0], obs_pts[1])
        sample[var1.data_name][sample[var1.data_name] == 0.0] = np.nan
        fig = ot.plot.line(sample, var1, var2, outer_legend=0.8)
        fig.tight_layout()
        return fig

    @pytest.mark.parametrize(
        ("var1", "var2"),
        [
            (ot.variables.pressure, ot.variables.temperature),
            (ot.variables.temperature, ot.variables.velocity),
        ],
    )
    @pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 30})
    def test_temporal_phaseplot(
        self, var1: ot.variables.Variable, var2: ot.variables.Variable
    ) -> plt.Figure:
        """Test plot.line over 2 vars temporally via image comparison."""
        ms = examples.load_meshseries_HT_2D_XDMF()
        ms_pts = ms.probe((0, 10, 0))
        fig = ot.plot.line(ms_pts, var1, var2, outer_legend=0.5)
        fig.tight_layout()
        return fig

    @pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 30})
    def test_lineplot_kwargs(self):
        """Test plot.line with possible kwargs via image comparison."""
        mesh = examples.load_meshseries_THM_2D_PVD().mesh(-1)
        obs_pts = np.reshape(mesh.center * 2, (2, 3))
        obs_pts[:, 0] = mesh.bounds[:2]
        sample = mesh.sample_over_line(obs_pts[0], obs_pts[1])
        _, ax = plt.subplots()
        with pytest.raises(TypeError):
            ot.plot.line(sample, ax)
        fig = ot.plot.line(
            sample, "x", ot.variables.temperature, sort=False, figsize=[10, 5],
            color="g", linewidth=1, ls="--", label="test", grid=True,
        )  # fmt: skip
        fig.tight_layout()
        return fig

    @pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 30})
    def test_lineplot_PETSC(self):
        """Test plot.line with PETSC results via image comparison."""
        ms = examples.load_meshseries_PETSc_2D(time_unit=("a", "a"))
        points_coords = np.array([[0.3, 0.5, 0.0], [0.24, 0.21, 0.0]])
        labels = [f"{label} linear interpolated" for label in ["pt0", "pt1"]]
        ms_pts = ms.probe(points_coords)
        var = ot.variables.pressure.replace(output_unit="mPa")
        fig = ot.plot.line(
            ms_pts, "time", var, labels=labels, colors=["b", "r"]
        )
        fig.tight_layout()
        return fig

    # ### Testing animation ####################################################

    @pytest.mark.parametrize(
        ("save", "ext", "writer", "err"),
        [
            (None, "", "", None),
            (FFMpegWriter.isAvailable(), ".mp4", "ffmpeg", None),
            (IMWriter.isAvailable(), ".gif", "ImageMagick", RuntimeWarning),
            (True, ".cpp", "", RuntimeError),
        ],
    )
    def test_animation(
        self, save: bool | None, ext: str, writer: str, err: Exception | None
    ):
        """Test creation and saving of an animation."""
        ms_full = examples.load_meshseries_THM_2D_PVD()
        timevalues = np.linspace(0, ms_full.timevalues[-1], num=3)
        ms = ms_full.resample_temporal(timevalues).transform(
            lambda mesh: mesh.clip("x")
        )
        fig = ot.plot.contourf(ms[0], ot.variables.temperature)

        def plot_func(timevalue: float, mesh: pv.UnstructuredGrid) -> None:
            fig.axes[0].clear()
            ot.plot.contourf(
                mesh, ot.variables.temperature, ax=fig.axes[0], dpi=50
            )
            fig.axes[0].set_title(f"{timevalue:.1f} yrs", fontsize=32)

        anim = ot.plot.animate(fig, plot_func, ms.timevalues, ms)
        if save is None:
            anim.to_jshtml()
        elif save:
            if err is RuntimeError:
                with pytest.raises(err):
                    utils.save_animation(anim, mkstemp(suffix=ext)[1], 5)
                anim.to_jshtml()
            elif err is RuntimeWarning:
                with pytest.warns(err):
                    utils.save_animation(anim, mkstemp(suffix=ext)[1], 5)
            else:
                utils.save_animation(anim, mkstemp(suffix=ext)[1], 5)
        else:
            pytest.skip(f"{writer} not available")
        plt.close()

    # ### Testing pyvista plots ################################################
    # TODO: we could use pytest-pyvista to do the same checks for the pyvista
    # plots as pyvista-mpl is doing for the matplotlib plots.

    @pytest.fixture()
    def pv_plotter(self):
        plotter = pv.Plotter(off_screen=True)
        yield plotter
        plotter.close()

    def test_pyvista_offscreen(self, pv_plotter: pv.Plotter):
        pv_plotter.add_mesh(pv.Sphere())
        pv_plotter.screenshot(filename=None)

    @pytest.mark.parametrize("opacities", [None, {7: 0.1, 10: 0.9}])
    def test_pv_plot(self, tmp_path: Path, opacities: dict | None):
        """Smoke test, test object: plot_contourf with pyvista

        Doesn't check for correctness of the plot itself"""
        mesh_3D = examples.load_meshseries_diffusion_3D()[-1]
        cpts = mesh_3D.cell_centers().points
        mesh_3D.cell_data["MaterialIDs"] = (12 * cpts[:, 0] + 3).astype(int)
        # test static plots
        ot.plot.contourf(
            mesh_3D, "MaterialIDs", opacities=opacities, interactive=False
        )
        # test dynamic plots with screenshot
        ot.plot.contourf(
            mesh_3D, "MaterialIDs", opacities=opacities
        ).screenshot(tmp_path / "test.png")
        assert (tmp_path / "test.png").is_file()

    def test_pv_plot_mask(self, tmp_path: Path):
        """Smoke test, test object: plot_contourf with pyvista / masked dataset

        Doesn't check for correctness of the plot itself"""
        mesh_3D = examples.load_mesh_mechanics_3D_cylinder()
        ot.plot.contourf(mesh_3D, "displacement").screenshot(
            tmp_path / "test.png"
        )
        assert (tmp_path / "test.png").is_file()
