"""Unit tests for meshlib."""

import textwrap
from pathlib import Path

import gmsh
import matplotlib.pyplot as plt
import numpy as np
import pytest
import pyvista as pv
from hypothesis import HealthCheck, Verbosity, example, given, settings
from hypothesis import strategies as st
from lxml import etree as ET

import ogstools as ot
from ogstools import examples
from ogstools.meshlib.meshes_from_yaml import meshes_from_yaml


def validate_msh_file(path: str) -> list[str]:
    gmsh.initialize()
    gmsh.open(path)

    gmsh.logger.start()
    messages = gmsh.logger.get()
    gmsh.logger.stop()

    gmsh.finalize()
    return messages


class TestUtils:
    """Test case for ogstools utilities."""

    def test_meshseries_xdmf(self):
        xdmf_ms = examples.load_meshseries_HT_2D_XDMF()._xdmf_reader
        assert xdmf_ms.has_fast_access()
        assert xdmf_ms.has_fast_access("temperature")
        assert xdmf_ms.rawdata_path("temperature").suffix == ".h5"
        assert xdmf_ms.rawdata_path().suffix == ".h5"

        xmf_ms = examples.load_meshseries_HT_2D_paraview_XMF()._xdmf_reader
        assert not xmf_ms.has_fast_access()
        assert not xmf_ms.has_fast_access("temperature")
        assert xmf_ms.rawdata_path("temperature").suffix in [".xdmf", ".xmf"]
        assert xmf_ms.rawdata_path().suffix in [".xdmf", ".xmf"]

    @pytest.mark.system()
    @pytest.mark.parametrize("quads", [True, False])
    def test_read_quadratic_xdmf(self, tmp_path, quads):
        "Test reading quadratic xdmf meshes. Tests the special case with a mesh with only 1 cell. Doesn't work with native meshio."
        mesh_path = tmp_path / "mesh.msh"

        ot.meshlib.rect(
            1, 1, structured_grid=quads, order=2, out_name=mesh_path
        )

        meshes = ot.Meshes.from_gmsh(mesh_path)
        meshes.save(tmp_path)

        model = ot.Project(
            output_file=tmp_path / "default.prj",
            input_file=examples.prj_mechanics,
        )
        model.replace_text("XDMF", xpath="./time_loop/output/type")
        model.replace_text(4, xpath=".//integration_order")
        model.write_input()
        model.run_model(write_logs=False, args=f"-m {tmp_path} -o {tmp_path}")
        ot.MeshSeries(tmp_path / "mesh_domain.xdmf").mesh(0)

    @pytest.mark.parametrize(
        "ht",
        [
            examples.load_meshseries_HT_2D_XDMF(),
            examples.load_meshseries_HT_2D_paraview_XMF(),
            examples.load_meshseries_HT_2D_PVD(),
            # examples.load_meshseries_HT_2D_VTU()
        ],
        ids=["XDMF", "XMF", "PVD"],
    )
    def test_meshseries_fileformats_indexing(self, ht):
        # all data is in one group in one h5 file
        num_timesteps = 97
        num_points = 190

        assert np.shape(ht.values("temperature")) == (num_timesteps, num_points)
        assert np.shape(ht.values("darcy_velocity")) == (
            num_timesteps,
            num_points,
            2,
        )
        assert np.shape(ht[7:].values("temperature")) == (
            num_timesteps - 7,
            num_points,
        )
        assert np.shape(ht[7:][::2].values("temperature")) == (
            (num_timesteps - 7) / 2,
            num_points,
        )
        assert np.shape(ht[7:][::2][5:].values("temperature")) == (
            (num_timesteps - 7) / 2 - 5,
            num_points,
        )
        # extra check here where mesh_cache is still empty
        ht_half = ht.transform(
            lambda mesh: mesh.clip("y", origin=mesh.center, crinkle=False)
        )
        assert np.shape(ht_half.values("temperature")) == (num_timesteps, 100)
        # artificial cell data
        for mesh in ht:
            mesh.cell_data["test"] = np.arange(mesh.n_cells)
        ht_half = ht.transform(
            lambda mesh: mesh.clip("y", origin=mesh.center, crinkle=True)
        )
        # indexing time and domain simultaneously
        assert ht_half[0].n_points == 100
        assert np.shape(ht_half[1:-1].values("temperature")) == (
            num_timesteps - 2,
            100,
        )
        assert ht_half[-1].n_cells == 81
        assert np.shape(ht_half[1:-1].values("test")) == (num_timesteps - 2, 81)
        # nested transform
        ht_quarter = ht_half.transform(
            lambda mesh: mesh.clip("y", origin=mesh.center, crinkle=False)
        )
        assert ht_quarter[0].n_cells == 45
        assert np.shape(ht_quarter.values("test")) == (num_timesteps, 45)
        assert np.shape(ht[1:3].values("temperature")) == (2, num_points)
        assert np.shape(ht[1]["temperature"]) == (num_points,)
        assert np.shape(ht[1:2].values("temperature")) == (1, num_points)
        assert ht.extract(slice(1, 4))[0].n_points == 3
        last_2_steps = ht[-2:].extract(slice(1, 4)).values("darcy_velocity")
        assert np.shape(last_2_steps) == (2, 3, 2)
        assert ht.extract(slice(1, 4), "cells")[0].n_cells == 3
        last_2_steps = ht[-2:].extract(slice(1, 4), "cells").values("test")
        assert np.shape(last_2_steps) == (2, 3)

    @pytest.mark.parametrize(
        "ht",
        [
            examples.load_meshseries_HT_2D_XDMF()[-2:-1],
            examples.load_meshseries_HT_2D_paraview_XMF()[-2:-1],
            examples.load_meshseries_HT_2D_PVD()[-2:-1],
            examples.load_meshseries_HT_2D_VTU(),
        ],
    )
    def test_meshseries_values(self, ht):
        assert np.shape(ht.values("temperature")) == (1, 190)
        assert np.shape(ht.values("darcy_velocity")) == (1, 190, 2)
        assert np.shape(ht.values("temperature")[:, 0:5]) == (1, 5)
        assert np.shape(ht.values("darcy_velocity")[:, 1:4, :]) == (1, 3, 2)

    def test_meshseries_xdmf_rawdata(self):
        xdmf = examples.load_meshseries_HT_2D_XDMF()
        h5file = xdmf.rawdata_file()
        assert h5file is not None
        assert h5file.suffix == ".h5"

        # This XDMF file is not generate via OGS/OGSTools, therefore the
        # underlying file structure is not known and no optimization is possible.
        xmf = examples.load_meshseries_HT_2D_paraview_XMF()
        assert xmf.rawdata_file() is None

    def test_all_types(self):
        pvd = examples.load_meshseries_THM_2D_PVD()
        xdmf = examples.load_meshseries_HT_2D_XDMF()
        with pytest.raises(TypeError, match="Can only read"):
            ot.MeshSeries(__file__)

        for ms in [pvd, xdmf]:
            try:
                mesh1 = ms.mesh(0)
                mesh1_closest = ms.mesh(ms.closest_timestep(1e-6))
            except Exception:
                pytest.fail("Read functions of MeshSeries failed")

            assert mesh1 == mesh1_closest
            assert not np.any(np.isnan(ms.timesteps))
            assert not np.any(np.isnan(ms.values("temperature")))

            assert ms.timevalues[
                ms.closest_timestep(1.0)
            ] == ms.closest_timevalue(1.0)

            ms.clear_cache()

    def test_items(self):
        ms = examples.load_meshseries_HT_2D_XDMF()
        for i, (timevalue, mesh) in enumerate(ms.items()):
            assert timevalue == ms.timevalues[i]
            assert mesh == ms[i]

    def test_cache(self):
        ms = examples.load_meshseries_HT_2D_PVD()
        assert not ms._is_all_cached  # pylint: disable=W0212
        _ = ms[::2]
        assert not ms._is_all_cached  # pylint: disable=W0212
        _ = ms.values("temperature")
        assert ms._is_all_cached  # pylint: disable=W0212

    def test_datadict(self):
        "Test getting and setting values inside the different data arrays"
        ms = examples.load_meshseries_THM_2D_PVD()
        np.testing.assert_array_equal(
            ms.values("temperature"), ms.point_data["temperature"]
        )
        ref_cell_values = 0.5 * ms.values("effective_pressure")
        ms.cell_data["effective_pressure"] *= 0.5
        np.testing.assert_array_equal(
            ref_cell_values, ms.cell_data["effective_pressure"]
        )
        assert np.all(ms.field_data["sigma_ip"] != 0.0)
        ms.field_data["sigma_ip"] = 0.0
        assert np.all(ms.field_data["sigma_ip"] == 0.0)

        for ms_key, mesh_key in zip(
            ms.point_data, ms[0].point_data, strict=True
        ):
            assert ms_key == mesh_key
        assert len(ms.cell_data) == len(ms[0].cell_data)

    def test_cache_copy(self):
        "Test that the cache of a MeshSeries is a deep copy as well."
        ms = examples.load_meshseries_HT_2D_XDMF()
        _ = ms[0]
        ms_subset = ms.copy().extract(0, "cells")
        assert ms[0].number_of_cells > ms_subset[0].number_of_cells

    def test_reindexing(self):
        "Test that indexing returns the correct meshes"
        ms = examples.load_meshseries_HT_2D_XDMF()
        ms_skip_first = ms[1:]
        for index_1, index_2 in [[0, 1], [-1, -1]]:
            for a, b in zip(
                ms_skip_first[index_1].point_data.values(),
                ms[index_2].point_data.values(),
                strict=True,
            ):
                np.testing.assert_array_equal(a, b)

    def test_time_aggregate(self):
        "Test aggregation of meshseries."
        mesh_series = examples.load_meshseries_HT_2D_XDMF()
        funcs = [np.min, np.max, np.mean, np.median, np.sum, np.std, np.var]
        for func in funcs:
            agg_mesh = mesh_series.aggregate_over_time("temperature", func)
            assert not np.any(
                np.isnan(agg_mesh["temperature_" + func.__name__])
            )

    def test_aggregate_time_of_limit(self):
        "Test aggregation of meshseries."
        mesh_series = examples.load_meshseries_HT_2D_XDMF()
        min_mesh = mesh_series.time_of_min("temperature")
        assert not np.any(np.isnan(min_mesh["min_temperature_time"]))
        max_mesh = mesh_series.time_of_max("temperature")
        assert not np.any(np.isnan(max_mesh["max_temperature_time"]))

    def test_time_aggregate_mesh_dependent(self):
        "Test aggregation of mesh_dependent variable on meshseries."
        mesh_series = examples.load_meshseries_THM_2D_PVD()
        prop = ot.variables.dilatancy_alkan
        agg_mesh = mesh_series.aggregate_over_time(prop, np.max)
        assert not np.any(np.isnan(agg_mesh[prop.output_name + "_max"]))
        agg_mesh = mesh_series.time_of_max(prop)
        assert not np.any(np.isnan(agg_mesh[f"max_{prop.output_name}_time"]))

    def test_domain_aggregate(self):
        "Test aggregation of meshseries."
        ms = examples.load_meshseries_HT_2D_XDMF()[1:]
        temp_min = ms.aggregate_over_domain("temperature", np.min)
        temp_mean = ms.aggregate_over_domain("temperature", np.mean)
        temp_max = ms.aggregate_over_domain("temperature", np.max)
        assert np.all(temp_max > temp_mean)
        assert np.all(temp_mean > temp_min)

    def test_plot_domain_aggregate(self):
        "Test aggregation of meshseries."
        mesh_series = examples.load_meshseries_THM_2D_PVD()
        _ = mesh_series.plot_domain_aggregate("temperature", np.mean)
        plt.close()

    def test_time_slice(self):
        results = examples.load_meshseries_HT_2D_XDMF()
        points = np.linspace([2, 2, 0], [4, 18, 0], num=100)
        ms_pts = ot.MeshSeries.extract_probe(results, points, "temperature")
        fig = ms_pts.plot_time_slice(
            "x", "time", "temperature", levels=[78, 79, 80]
        )
        fig = ms_pts.plot_time_slice(
            "time", "y", "temperature", time_logscale=True, cb_loc="left",
            dpi=50, fontsize=10
        )  # fmt: skip
        with pytest.raises(ValueError, match="fig and ax together"):
            _ = ms_pts.plot_time_slice("x", "time", "temperature", fig=fig)
        with pytest.raises(KeyError, match="has to be 'time'"):
            _ = ms_pts.plot_time_slice("x", "y", "temperature")
        with pytest.raises(KeyError, match="has to be a spatial"):
            _ = ms_pts.plot_time_slice("time", "temperature", "y")

        plt.close()

    def _check_probe(self, ms: ot.MeshSeries, points: np.ndarray) -> None:
        "checks different variants for points arg: point_array, point, [point]"
        pt_ids = np.asarray([ms[0].find_closest_point(pt) for pt in points])
        ref_values = np.asarray([mesh["temperature"][pt_ids] for mesh in ms])
        slice_shapes = [
            [slice(None), (len(ms.timevalues), len(points))],
            [0, (len(ms.timevalues),)],
            [[0], (len(ms.timevalues), 1)],
        ]
        for method in ["nearest", "linear"]:
            for slicing, shape in slice_shapes:
                values = ms.probe(points[slicing], "temperature", method)
                assert values.shape == shape
                np.testing.assert_allclose(values, ref_values[:, slicing])
                assert not np.any(np.isnan(values))

    def test_probe_2D_mesh_PETSC(self):
        ms = examples.load_meshseries_PETSc_2D()
        ms.scale(time=("s", "a"))
        points_coords = np.array([[0.3, 0.5, 0.0], [0.24, 0.21, 0.0]])
        labels = [f"{label} linear interpolated" for label in ["pt0", "pt1"]]
        ms_pts = ot.MeshSeries.extract_probe(ms, points_coords)
        ot.plot.line(
            ms_pts,
            "time",
            ot.variables.pressure,
            labels=labels,
            colors=["b", "r"],
        )

    def test_probe_2D_mesh(self):
        "Test point probing on a 2D meshseries."
        ms = examples.load_meshseries_HT_2D_XDMF()
        pt_min, pt_max = np.reshape(ms[0].bounds, (3, 2)).T
        points = np.linspace(pt_min, pt_max, num=10)
        self._check_probe(ms, points)

    def test_probe_1D_mesh(self):
        "Test point probing on a 1D meshseries."
        ms = examples.load_meshseries_HT_2D_XDMF()
        ms_1D = ms.extract(ms[0].points[:, 1] == 0)
        pt_min, pt_max = np.reshape(ms_1D[0].bounds, (3, 2)).T
        points = np.linspace(pt_min, pt_max, num=10)
        self._check_probe(ms_1D, points)

    def test_extract_probe_1D_mesh_single_pt(self):
        "Test single point probing on a 1D meshseries."
        ms = examples.load_meshseries_HT_2D_XDMF()
        ms_1D = ms.extract(ms[0].points[:, 1] == 0)
        pt_min = np.reshape(ms_1D[0].bounds, (3, 2)).T[0]
        ms_ref = ot.MeshSeries.extract_probe(ms_1D, pt_min)
        for key in ms_ref.point_data:
            np.testing.assert_array_equal(
                ms_ref.values(key)[:, 0], ms_1D.values(key)[:, 0]
            )

    def test_probe_multiple(self):
        """Test probe with mixed scalar, vector point and cell data.

        Checks that extracting all vars at once produces the same data fields as
        you would get by extract the different data fields individually.
        """
        ms = examples.load_meshseries_THM_2D_PVD()
        pt_min, pt_max = np.reshape(ms[0].bounds, (3, 2)).T
        points = np.linspace(pt_min, pt_max, num=10, endpoint=False)
        custom_keys = [
            ot.variables.stress["xx"],
            ot.variables.stress.von_Mises,
            "displacement",
            "MaterialIDs",
        ]
        all_keys = set().union(ms.point_data.keys(), ms.cell_data.keys())
        for arg, keys in [(custom_keys, custom_keys), (None, all_keys)]:
            ms_pts = ot.MeshSeries.extract_probe(ms, points, arg)
            for key in keys:
                ms_ref = ot.MeshSeries.extract_probe(ms, points, key)
                np.testing.assert_array_equal(
                    ms_pts.values(key), ms_ref.values(key)
                )

    def test_plot_probe(self):
        """Test creation of probe plots."""
        ms = examples.load_meshseries_THM_2D_PVD()
        ms_pts = ot.MeshSeries.extract_probe(ms, np.array(ms.mesh(0).center))
        _ = ot.plot.line(ms_pts, "time", ot.variables.temperature)
        ms_pts = ot.MeshSeries.extract_probe(ms, ms.mesh(0).points[[0, -1]])
        _ = ot.plot.line(ms_pts, "time", ot.variables.temperature)
        _ = ot.plot.line(ms_pts, "time", ot.variables.velocity)
        _ = ot.plot.line(ms_pts, "time", ot.variables.stress)
        _ = ot.plot.line(ms_pts, "time", ot.variables.stress.von_Mises)
        ms = examples.load_meshseries_HT_2D_XDMF()
        ms_pts = ot.MeshSeries.extract_probe(ms, np.array(ms.mesh(0).center))
        _ = ot.plot.line(ms_pts, "time", ot.variables.temperature)
        _ = ot.plot.line(ms_pts, "time", ot.variables.velocity)
        plt.close()

    def test_extract_probe(self):
        results = examples.load_meshseries_HT_2D_XDMF()
        points = np.linspace([2, 2, 0], [4, 18, 0], num=100)
        ms_pts = ot.MeshSeries.extract_probe(results, points, "temperature")
        np.testing.assert_array_equal(ms_pts[0].points, points)
        ms_pts = ot.MeshSeries.extract_probe(
            results, points, "temperature", "nearest"
        )
        pt_id = results[0].find_closest_point(points[0])
        np.testing.assert_array_equal(
            ms_pts["temperature"][:, 0], results["temperature"][:, pt_id]
        )

    def test_resample(self):
        results = examples.load_meshseries_HT_2D_XDMF()
        in_between = 0.5 * (results.timevalues[:-1] + results.timevalues[1:])
        resampled = ot.MeshSeries.resample(results, in_between)
        for idx, mesh in enumerate(resampled):
            for var in ["temperature", "pressure", "darcy_velocity"]:
                delta = results[idx + 1][var] - results[idx][var]
                half_delta = mesh[var] - results[idx][var]
                np.testing.assert_almost_equal(half_delta, 0.5 * delta)

    def test_diff_two_meshes(self):
        meshseries = examples.load_meshseries_THM_2D_PVD()
        mesh1 = meshseries.mesh(0)
        mesh2 = meshseries.mesh(-1)
        mesh_diff = ot.meshlib.difference(mesh1, mesh2, "temperature")
        # test, that no sampling occurs for equal topology
        np.testing.assert_array_equal(
            mesh_diff["temperature_difference"],
            mesh1["temperature"] - mesh2["temperature"],
        )
        # test same/different topology and scalar / vector variable
        for scaling in [1.0, 2.0]:
            for variable in ["temperature", "velocity"]:
                mesh_diff = ot.meshlib.difference(
                    mesh1.scale(scaling), mesh2, variable
                )

        quad_tri_diff = ot.meshlib.difference(
            mesh1.triangulate(), mesh1, "temperature"
        )
        quad_tri_diff_vals = ot.variables.temperature.difference.transform(
            quad_tri_diff
        )
        np.testing.assert_allclose(quad_tri_diff_vals, 0.0, atol=1e-12)
        mesh_diff = ot.meshlib.difference(
            mesh1, mesh2, ot.variables.temperature
        )
        assert isinstance(mesh_diff, pv.UnstructuredGrid)
        mesh_diff = ot.meshlib.difference(mesh1, mesh2)

    def test_diff_pairwise(self):
        n = 5
        meshseries = examples.load_meshseries_THM_2D_PVD()
        meshes1 = [meshseries.mesh(0)] * n
        meshes2 = [meshseries.mesh(-1)] * n
        meshes_diff = ot.meshlib.difference_pairwise(
            meshes1, meshes2, ot.variables.temperature
        )
        assert isinstance(meshes_diff, np.ndarray)
        assert len(meshes_diff) == n

        meshes_diff = ot.meshlib.difference_pairwise(meshes1, meshes2)

    def test_diff_matrix_single(self):
        meshseries = examples.load_meshseries_THM_2D_PVD()
        meshes1 = [meshseries.mesh(0), meshseries.mesh(-1)]
        meshes_diff = ot.meshlib.difference_matrix(
            meshes1, variable=ot.variables.temperature
        )
        assert isinstance(meshes_diff, np.ndarray)

        assert meshes_diff.shape == (
            len(meshes1),
            len(meshes1),
        )

        meshes_diff = ot.meshlib.difference_matrix(meshes1)

    def test_diff_matrix_unequal(self):
        meshseries = examples.load_meshseries_THM_2D_PVD()
        meshes1 = [meshseries.mesh(0), meshseries.mesh(-1)]
        meshes2 = [meshseries.mesh(0), meshseries.mesh(-1), meshseries.mesh(-1)]
        meshes_diff = ot.meshlib.difference_matrix(
            meshes1, meshes2, ot.variables.temperature
        )
        assert isinstance(meshes_diff, np.ndarray)
        assert meshes_diff.shape == (
            len(meshes1),
            len(meshes2),
        )
        meshes_diff = ot.meshlib.difference_matrix(meshes1, meshes2)

    @pytest.mark.parametrize(
        "mesh",
        [
            examples.load_mesh_mechanics_2D(),
            ot.Mesh(pv.SolidSphere(100, center=(0, 0, -101))),
        ],
    )
    def test_depth(self, mesh: ot.Mesh):
        mesh = examples.load_mesh_mechanics_2D()
        mesh["depth"] = mesh.depth(use_coords=True)
        depth_idx = 2 if mesh.volume != 0.0 else 1
        assert np.all(mesh["depth"] == -mesh.points[..., depth_idx])
        mesh["depth"] = mesh.depth()
        assert np.all(mesh["depth"] < -mesh.points[..., depth_idx])

    @pytest.mark.parametrize(
        ("elem_order", "quads", "intpt_order", "mixed"),
        [
            (1, False, 2, False),
            (1, False, 2, True),
            (1, False, 3, False),
            (1, False, 4, False),
            (1, True, 2, False),
            (1, True, 3, False),
            (1, True, 4, False),
            (2, False, 4, False),
            (2, True, 4, False),
        ],
    )
    @pytest.mark.system()
    def test_ip_mesh(self, tmp_path, elem_order, quads, intpt_order, mixed):
        "Test creation of integration point meshes."

        mesh_path = tmp_path / "mesh.msh"
        sigma_ip = ot.variables.stress.replace(data_name="sigma_ip")

        ot.meshlib.rect(
            n_edge_cells=6,
            n_layers=2,
            structured_grid=quads,
            order=elem_order,
            out_name=mesh_path,
            mixed_elements=mixed,
            jiggle=0.01,
        )
        meshes = ot.Meshes.from_gmsh(mesh_path)
        meshes.save(tmp_path)

        model = ot.Project(
            output_file=tmp_path / "default.prj",
            input_file=examples.prj_mechanics,
        )
        model.replace_text(intpt_order, xpath=".//integration_order")
        model.write_input()
        model.run_model(write_logs=True, args=f"-m {tmp_path} -o {tmp_path}")
        meshseries = ot.MeshSeries(tmp_path / "mesh.pvd")
        int_pts = meshseries.mesh(-1).to_ip_point_cloud()
        ip_ms = meshseries.ip_tesselated()
        ip_mesh = ip_ms.mesh(-1)
        vals = ip_ms.probe(ip_mesh.center, sigma_ip.data_name)
        assert not np.any(np.isnan(vals))
        assert int_pts.number_of_points == ip_mesh.number_of_cells
        containing_cells = ip_mesh.find_containing_cell(int_pts.points)
        # check for integration points coinciding with the tessellated cells
        np.testing.assert_equal(
            sigma_ip.magnitude.transform(ip_mesh)[containing_cells],
            sigma_ip.magnitude.transform(int_pts),
        )

    def test_reader(self):
        assert isinstance(examples.load_meshseries_THM_2D_PVD(), ot.MeshSeries)
        assert isinstance(ot.MeshSeries(examples.elder_xdmf), ot.MeshSeries)
        assert isinstance(ot.Mesh.read(examples.mechanics_2D), ot.Mesh)

    @pytest.mark.system()
    def test_xdmf_quadratic(self, tmp_path):
        "Test reading of quadratic elements in xdmf."

        msh_path = tmp_path / "mesh.msh"
        ot.meshlib.rect(
            n_edge_cells=6, structured_grid=False, order=2, out_name=msh_path
        )

        meshes = ot.Meshes.from_gmsh(msh_path)
        meshes.save(tmp_path)

        model = ot.Project(
            input_file=examples.prj_mechanics,
            output_file=tmp_path / "default.prj",
        )
        model.replace_text("4", xpath=".//integration_order")
        model.replace_text("XDMF", xpath="./time_loop/output/type")
        model.write_input()
        model.run_model(write_logs=True, args=f"-m {tmp_path} -o {tmp_path}")
        mesh = ot.MeshSeries(tmp_path / "mesh_domain.xdmf").mesh(-1)
        assert not np.any(np.isnan(ot.variables.stress.transform(mesh)))

    def test_remesh_with_tri(self, tmp_path):
        mesh = examples.load_meshseries_THM_2D_PVD().mesh(1)
        msh_path = tmp_path / "tri_mesh.msh"
        ot.meshlib.gmsh_meshing.remesh_with_triangles(mesh, msh_path)
        assert len(
            ot.Meshes.from_gmsh(msh_path, reindex=False, log=False)
        ) == 1 + len(np.unique(mesh["MaterialIDs"]))
        # boundaries are not assigned a physical tag in remesh_with_trinagles

    def test_indexing(self):
        ms = examples.load_meshseries_HT_2D_XDMF()
        assert isinstance(ms[1], ot.Mesh)

    def test_slice(self):
        ms = examples.load_meshseries_HT_2D_XDMF()
        ms_sliced = ms[1::2]
        assert len(ms.timevalues) >= 2 * len(ms_sliced.timevalues)

    def test_transform(self):
        ms = examples.load_meshseries_THM_2D_PVD()
        ms_mod = ms.transform(lambda mesh: mesh.slice("x"))
        assert max(ms[0].cells) == 3779  # Check if example mesh has changed
        assert max(ms_mod[0].cells) == 44
        assert len(ms[0].points) == 3780  # Check if example mesh has changed
        assert len(ms_mod[0].points) == 45

    def test_copy_deep(self):
        ms = examples.load_meshseries_THM_2D_PVD()
        ms.point_data["temperature"] = 0
        ms_deepcopy = ms.copy(deep=True)
        ms_deepcopy.point_data["temperature"] = 1
        assert np.all(
            ms.point_data["temperature"]
            != ms_deepcopy.point_data["temperature"]
        )

    def test_copy_shallow(self):
        ms = examples.load_meshseries_THM_2D_PVD()
        ms.point_data["temperature"] = 0
        ms_shallowcopy = ms.copy(deep=False)
        ms_shallowcopy.point_data["temperature"] = 1
        assert np.all(
            ms.point_data["temperature"]
            == ms_shallowcopy.point_data["temperature"]
        )

    def test_save_pvd_mesh_series(self, tmp_path):
        file_name = "test.pvd"

        ms = examples.load_meshseries_HT_2D_PVD()
        ms.save(tmp_path / file_name, deep=True)
        ms_test = ot.MeshSeries(tmp_path / file_name)
        assert len(ms.timevalues) == len(ms_test.timevalues)
        assert np.abs(ms.timevalues[1] - ms_test.timevalues[1]) < 1e-14
        for var in ["temperature", "darcy_velocity", "pressure"]:
            val_ref = np.sum(ms.aggregate_over_domain(var, np.max))
            val_test = np.sum(ms_test.aggregate_over_domain(var, np.max))
            assert np.abs(val_ref - val_test) < 1e-14

        for m in ms_test:
            assert "test" in m.filepath.name

        # Smoke test for ascii output
        ms.save(tmp_path / "test_ascii.pvd", ascii=True)

        ms.save(tmp_path / file_name, deep=False)
        tree = ET.parse(tmp_path / file_name)
        num_slices = len(ms.timevalues)
        num_slices_test = len(tree.findall("./Collection/DataSet"))
        assert num_slices == num_slices_test
        pvd_entries = tree.findall("./Collection/DataSet")
        for i in range(num_slices):
            assert ms[i].filepath.name == pvd_entries[i].attrib["file"]
            ts = float(pvd_entries[i].attrib["timestep"])
            assert np.abs(ms.timevalues[i] - ts) < 1e-14

    def test_save_xdmf_mesh_series(self, tmp_path):
        file_name = "test.pvd"

        ms = examples.load_meshseries_CT_2D_XDMF()
        ms.save(tmp_path / file_name, deep=True)
        ms_test = ot.MeshSeries(tmp_path / file_name)
        assert len(ms.timevalues) == len(ms_test.timevalues)
        assert np.abs(ms.timevalues[1] - ms_test.timevalues[1]) < 1e-14
        assert (
            np.abs(
                np.sum(ms.aggregate_over_domain("Si", np.max))
                - np.sum(ms_test.aggregate_over_domain("Si", np.max))
            )
            < 1e-14
        )
        for m in ms_test:
            assert "test" in m.filepath.name

        ms.save(tmp_path / file_name, deep=False)
        tree = ET.parse(tmp_path / file_name)
        num_slices = len(ms.timevalues)
        pvd_entries = tree.findall("./Collection/DataSet")
        num_slices_test = len(pvd_entries)
        assert num_slices == num_slices_test
        for i in range(num_slices):
            ts = float(pvd_entries[i].attrib["timestep"])
            assert np.abs(ms.timevalues[i] - ts) < 1e-14

    def test_remove_array(self):

        def data(ms_or_mesh: ot.MeshSeries) -> tuple[tuple[dict, str, int]]:
            "return a tuple of datafields, array_to_remove and num_arrays"
            return (
                (ms_or_mesh.point_data, "temperature", 12),
                (ms_or_mesh.cell_data, "effective_pressure", 5),
                (ms_or_mesh.field_data, "sigma_ip", 4),
            )

        for slicing in [slice(None), slice(-1)]:
            ms = examples.load_meshseries_THM_2D_PVD()
            for datafield, arr_to_rm, num_arrays in data(ms[slicing]):
                assert arr_to_rm in datafield, "Expected array is missing."
                assert (
                    len(datafield.keys()) == num_arrays
                ), "Unexpected number of arrays in MeshSeries."
                del datafield[arr_to_rm]
                assert arr_to_rm not in datafield, "Deleted array still exists."
                assert (
                    len(datafield.keys()) == num_arrays - 1
                ), "Unexpected number of arrays after deletion."
            for datafield, arr_to_rm, num_arrays in data(ms.mesh(-1)):
                skip_last = slicing == slice(-1)
                array_present = arr_to_rm in datafield
                assert array_present == skip_last
                expected_num_arrays = num_arrays - int(not skip_last)
                assert len(datafield.keys()) == expected_num_arrays

    @pytest.mark.parametrize(
        "load_example_ms",
        [
            examples.load_meshseries_HT_2D_PVD,
            examples.load_meshseries_HT_2D_XDMF,
        ],
    )
    def test_extend(self, load_example_ms):
        ms1 = load_example_ms()
        len_orig_ms1 = len(ms1)
        ms1_orig_tv = ms1.timevalues
        ms1.extend(ms1)
        print(ms1)  # check if representable
        assert len(ms1) == 2 * len_orig_ms1 - 1
        assert ms1.timevalues[-1] == ms1_orig_tv[-1] * 2
        ms1 = load_example_ms()
        ms2 = load_example_ms()
        ms2._timevalues = ms1.timevalues[-1] + ms2.timevalues
        ms1.extend(ms2)
        assert len(ms1) == 2 * len_orig_ms1 - 1
        assert ms1.timevalues[-1] == ms1_orig_tv[-1] * 2
        ms1 = load_example_ms()
        ms2 = load_example_ms()
        ms2._timevalues = ms1_orig_tv[-1] + ms1_orig_tv + 0.1
        ms1.extend(ms2)
        assert len(ms1) == 2 * len_orig_ms1
        assert ms1.timevalues[-1] == ms1_orig_tv[-1] * 2 + 0.1

    def test_reshape_obs_points(self):
        points_x = (1,)
        pts_x = ot.meshlib._utils.reshape_obs_points(points_x)
        assert pts_x.shape == (1, 3)

    def test_reshape_obs_points_mesh(self):
        ms = examples.load_meshseries_CT_2D_XDMF()
        mesh = ms.mesh(0)
        points = ((-150.0, 75.0), (-147.65625, 72.65625))
        pts = ot.meshlib._utils.reshape_obs_points(points, mesh)
        np.testing.assert_equal(
            pts, np.asarray([[-150, 0, 75], [-147.65625, 0, 72.65625]])
        )

    def test_ms_active(self):
        # This test checks if mask is applied correctly
        # when difference is computed.
        ms = examples.load_meshseries_THM_2D_PVD()
        # For this test to run, loaded data set needs to
        # contain fields: pressure, pressure_active and
        # temperature but not temperature active
        assert ot.variables.pressure.data_name in ms.point_data
        assert ot.variables.pressure.mask in ms.cell_data
        assert ot.variables.temperature.data_name in ms.point_data
        assert ot.variables.temperature.mask not in ms.cell_data
        m_diff_T = ms[-1].difference(ms[0], ot.variables.temperature)
        m_diff_p = ms[-1].difference(ms[0], ot.variables.pressure)
        assert (
            np.count_nonzero(
                np.isnan(
                    m_diff_T[ot.variables.temperature.difference.data_name]
                )
            )
            == 0
        )
        assert (
            np.count_nonzero(
                np.isnan(m_diff_p[ot.variables.pressure.difference.data_name])
            )
            != 0
        )

    @pytest.mark.parametrize("threshold_angle", [None, 20.0])
    @pytest.mark.parametrize("angle_y", [0.0, -20.0, -45.0, -70.0])
    def test_meshes_from_mesh(
        self, threshold_angle: None | float, angle_y: float
    ):
        mesh = examples.load_meshseries_THM_2D_PVD()[0].rotate_y(angle_y)
        boundaries = ot.Meshes.from_mesh(mesh, threshold_angle)
        # not recommended, but here we are only interested in the boundaries
        boundaries.pop("domain")

        assert len(boundaries) == 4
        np.testing.assert_array_equal(
            [mesh.n_cells for mesh in boundaries.values()], [83, 83, 44, 44]
        )
        assert np.all([mesh.n_cells > 1 for mesh in boundaries.values()])
        assert np.all(boundaries["left"].points[:, 0] == mesh.bounds[0])
        assert np.all(boundaries["right"].points[:, 0] == mesh.bounds[1])
        assert boundaries["bottom"].bounds[2] == mesh.bounds[2]
        assert boundaries["top"].bounds[3] == mesh.bounds[3]

    @pytest.mark.system()
    def test_meshes_from_mesh_run(self, tmp_path):
        "Test using extracted boundaries for a simulation."
        mesh_path = tmp_path / "mesh.msh"
        ot.meshlib.rect(n_edge_cells=(2, 4), out_name=mesh_path)
        domain = ot.Meshes.from_gmsh(mesh_path)["domain"]
        # this is no good practice and only done for testing purposes
        # we recommend to define the boundaries as physical groups within gmsh
        meshes = ot.Meshes.from_mesh(domain)
        meshes.save(tmp_path)

        model = ot.Project(
            input_file=examples.prj_mechanics,
            output_file=tmp_path / "default.prj",
        )
        model.write_input()
        model.run_model(write_logs=False, args=f"-m {tmp_path} -o {tmp_path}")
        # check for correct bulk id mapping during extraction
        mesh = ot.MeshSeries(tmp_path / "mesh.pvd")[-1]
        top_right = mesh.find_closest_point([1.0, 1.0, 0.0])
        max_uy = np.max(mesh["displacement"][:, 1])
        assert max_uy == mesh["displacement"][top_right, 1]

    @st.composite
    def meshing(draw: st.DrawFn):
        dim = draw(st.integers(min_value=2, max_value=3))
        mesh_func = ot.meshlib.rect if dim == 2 else ot.meshlib.cuboid
        n_cells = draw(st.tuples(*([st.integers(1, 3)] * dim)))
        n_layers = draw(st.integers(min_value=1, max_value=4))
        rand_id = draw(st.integers(min_value=0, max_value=n_layers - 1))
        return mesh_func, n_cells, n_layers, rand_id

    @pytest.mark.tools()
    @example(meshing_data=(ot.meshlib.rect, (2, 2), 2, 0), failcase=True).xfail(
        # CLI version fails and doesn't write the new file, thus cannot be read
        raises=FileNotFoundError
    )
    @given(meshing_data=meshing(), failcase=st.just(False))
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        verbosity=Verbosity.normal,
        deadline=None,
    )
    def test_identify_subdomains(self, tmp_path, meshing_data, failcase):
        "Testing parity between py and C++ (CLI) implementation."

        mesh_func, n_cells, n_layers, rand_id = meshing_data
        path = Path(tmp_path / f"{mesh_func.__name__}_{n_layers}_{n_cells}")
        path.mkdir(parents=True, exist_ok=True)
        mesh_name = path / "rect.msh"
        mesh_func(n_edge_cells=n_cells, n_layers=n_layers, out_name=mesh_name)
        meshes = ot.Meshes.from_gmsh(mesh_name, log=False)
        layer: pv.UnstructuredGrid
        layer = meshes["domain"].threshold([rand_id, rand_id], "MaterialIDs")
        # multi-dim test
        if meshes["domain"].volume:
            meshes["layer_surface"] = layer.extract_surface()
        meshes["layer_edges"] = layer.extract_feature_edges()
        meshes["layer_points"] = layer.extract_points(
            range(layer.n_points), include_cells=False
        )
        if failcase:
            meshes["extra_mesh"] = meshes["layer_points"].translate([-1, 0, 0])
        for m in meshes.values():
            m.point_data.pop("bulk_node_ids", None)
            m.cell_data.pop("bulk_element_ids", None)

        sub_paths = meshes.save(path, overwrite=True)
        domain_mesh = meshes.domain()
        ot.cli().identifySubdomains(
            f=True,
            o=path / "new_",
            m=domain_mesh.filepath,
            *sub_paths,  # noqa: B026
        )
        # actually meshes.subdomains(), but here we let the domain mesh also get bulk ids
        ot.meshlib.subdomains.identify_subdomains(domain_mesh, meshes.values())

        def _check(mesh_1, mesh_2, key: str):
            np.testing.assert_array_equal(mesh_1[key], mesh_2[key])

        for name, mesh in meshes.items():
            cli_subdomain = pv.read(path / f"new_{name}.vtu")
            _check(mesh, cli_subdomain, "bulk_node_ids")
            if "number_bulk_elements" in mesh.cell_data:
                _check(mesh, cli_subdomain, "number_bulk_elements")
                # order of the bulk element ids is random for multiple ids per
                # sub cell, thus only testing for equality of the set.
                cell_id = 0
                for num_cells in mesh["number_bulk_elements"]:
                    view = slice(cell_id, cell_id + num_cells)
                    np.testing.assert_array_equal(
                        set(mesh["bulk_element_ids"][view]),
                        set(cli_subdomain["bulk_element_ids"][view]),
                    )
                    cell_id += num_cells

            else:
                _check(mesh, cli_subdomain, "bulk_element_ids")

    def test_mfy_meshes_from_yaml(self, tmp_path):
        yaml_content = textwrap.dedent(
            """\
        parameters: {}
        points:
          p0: { coords: [0.0, 0.0], char_length: 0.1 }
          p1: { coords: [1.0, 0.0], char_length: 0.1 }
          p2: { coords: [1.0, 1.0], char_length: 0.1 }
          p3: { coords: [0.0, 1.0], char_length: 0.1 }
        lines:
          l0: { start: p0, end: p1 }
          l1: { start: p1, end: p2 }
          l2: { start: p2, end: p3 }
          l3: { start: p3, end: p0 }
        surfaces:
          s0:
            loop: [l0, l1, l2, l3]
        groups:
          Domain:
            dim: 2
            entities: [s0]
        """
        )

        yaml_file = tmp_path / "geom.yml"
        yaml_file.write_text(yaml_content)

        msh_file = meshes_from_yaml(yaml_file, tmp_path)

        print(f"mesh-file: {msh_file}")

        # prüfen, dass .msh erzeugt wurde
        assert msh_file.exists()
        assert msh_file.suffix == ".msh"

        issues = validate_msh_file(str(msh_file))
        assert not issues, f"Mesh validation failed: {issues}"

    def test_mfy_with_parameters(self, tmp_path):
        yaml_content = textwrap.dedent(
            """\
        parameters:
          a: 2
          b: "a * 3"
        points:
          p0: { coords: [0.0, "b"], char_length: "sqrt(4)" }
        lines: {}
        surfaces: {}
        groups: {}
        """
        )
        yaml_file = tmp_path / "geom.yml"
        yaml_file.write_text(yaml_content)
        msh_file = meshes_from_yaml(yaml_file, tmp_path)
        assert msh_file.exists()

    def test_mfy_with_list_coords(self, tmp_path):
        yaml_content = textwrap.dedent(
            """\
        parameters: {}
        points:
          p0: { coords: ["1+1", 2], char_length: 0.1 }
        lines: {}
        surfaces: {}
        groups: {}
        """
        )
        yaml_file = tmp_path / "geom.yml"
        yaml_file.write_text(yaml_content)
        msh_file = meshes_from_yaml(yaml_file, tmp_path)
        assert msh_file.exists()

    def test_mfy_invalid_type(self, tmp_path):
        yaml_content = textwrap.dedent(
            """\
        parameters: {}
        points:
          p0: { coords: [{foo: bar}, 0], char_length: 0.1 }
        lines: {}
        surfaces: {}
        groups: {}
        """
        )
        yaml_file = tmp_path / "geom.yml"
        yaml_file.write_text(yaml_content)

        with pytest.raises(TypeError, match="Unsupported type"):
            meshes_from_yaml(yaml_file, tmp_path)

    def test_mfy_invalid_expression(self, tmp_path):
        yaml_content = textwrap.dedent(
            """\
        parameters:
          bad: "foo + bar"
        points:
          p0: { coords: [0, 0], char_length: "bad" }
        lines: {}
        surfaces: {}
        groups: {}
        """
        )
        yaml_file = tmp_path / "geom.yml"
        yaml_file.write_text(yaml_content)
        with pytest.raises(ValueError, match="Failed to evaluate parameter"):
            meshes_from_yaml(yaml_file, tmp_path)

    def test_mfy_point_without_coords(self, tmp_path):
        yaml_content = textwrap.dedent(
            """\
        parameters: {}
        points:
          p0: { char_length: 0.1 }
        lines: {}
        surfaces: {}
        groups: {}
        """
        )
        yaml_file = tmp_path / "geom.yml"
        yaml_file.write_text(yaml_content)
        with pytest.raises(ValueError, match="Incomplete point definition"):
            meshes_from_yaml(yaml_file, tmp_path)

    def test_mfy_surface_without_loops(self, tmp_path):
        yaml_content = textwrap.dedent(
            """\
        parameters: {}
        points:
          p0: { coords: [0.0, 0.0], char_length: 0.1 }
          p1: { coords: [1.0, 0.0], char_length: 0.1 }
        lines:
          l0: { start: p0, end: p1 }
        surfaces:
          s0: {}
        groups:
          Domain:
            dim: 2
            entities: [s0]
        """
        )
        yaml_file = tmp_path / "geom.yml"
        yaml_file.write_text(yaml_content)
        with pytest.raises(ValueError, match="has no 'loop' or 'loops'"):
            meshes_from_yaml(yaml_file, tmp_path)

    def test_mfy_unsupported_group_dim(self, tmp_path):
        yaml_content = textwrap.dedent(
            """\
        parameters: {}
        points:
          p0: { coords: [0,0], char_length: 0.1 }
          p1: { coords: [1,0], char_length: 0.1 }
        lines:
          l0: { start: p0, end: p1 }
        surfaces: {}
        groups:
          TestGroup:
            dim: 3
            entities: [l0]
        """
        )
        yaml_file = tmp_path / "geom.yml"
        yaml_file.write_text(yaml_content)
        with pytest.raises(
            NotImplementedError, match="Unsupported group dimension"
        ):
            meshes_from_yaml(yaml_file, tmp_path)

    def test_mfy_arc_line(self, tmp_path):
        yaml_content = textwrap.dedent(
            """\
        parameters: {}
        points:
          p0: { coords: [1,0], char_length: 0.1 }
          p1: { coords: [0,-1], char_length: 0.1 }
          c0: { coords: [0,0], char_length: 0.1 }
        lines:
          a0: { start: p0, end: p1, center: c0 }
        surfaces: {}
        groups: {}
        """
        )
        yaml_file = tmp_path / "geom.yml"
        yaml_file.write_text(yaml_content)
        msh_file = meshes_from_yaml(yaml_file, tmp_path)
        assert msh_file.exists()

    def test_mfy_empty_blocks(self, tmp_path):
        yaml_content = textwrap.dedent(
            """\
        parameters: {}
        points: {}
        lines: {}
        surfaces: {}
        groups: {}
        """
        )
        yaml_file = tmp_path / "geom.yml"
        yaml_file.write_text(yaml_content)
        msh_file = meshes_from_yaml(yaml_file, tmp_path)
        assert msh_file.exists()

    def test_mfy_radioactive(self, tmp_path):
        yaml_content = textwrap.dedent(
            """\
        parameters:
          radius_dot: 2.0
          radius_inner: 2.5
          radius_outer: 6.5
          phi_0 : 0
          phi_60 : 1.04720
          phi_120 : 2.09440
          phi_180 : 3.14159
          phi_240 : 4.18879
          phi_300 : 5.23599
          box_size : "radius_outer * 2.2"
          center : "box_size / 2"

        points:

          b00: { coords: [0.0, 0.0], char_length: 0.3 }
          b01: { coords: ["box_size", 0.0], char_length: 0.3 }
          b02: { coords: ["box_size", "box_size"], char_length: 0.3 }
          b03: { coords: [0.0, "box_size"], char_length: 0.3 }

          p00: { coords: ["center", "center"], char_length: 0.3 }
          p01: { coords: ["radius_dot+center", "center"], char_length: 0.3 }
          p02: { coords: ["center-radius_dot", "center"], char_length: 0.3 }
          p03: { coords: ["(cos(phi_0) * radius_inner)+center", "(sin(phi_0) * radius_inner)+center"], char_length: 0.3 }
          p04: { coords: ["(cos(phi_0) * radius_outer)+center", "(sin(phi_0) * radius_outer)+center"], char_length: 0.3 }
          p05: { coords: ["(cos(phi_60) * radius_outer)+center", "(sin(phi_60) * radius_outer)+center"], char_length: 0.3 }
          p06: { coords: ["(cos(phi_60) * radius_inner)+center", "(sin(phi_60) * radius_inner)+center"], char_length: 0.3 }

          p07: { coords: ["(cos(phi_120) * radius_inner)+center", "(sin(phi_120) * radius_inner)+center"], char_length: 0.3 }
          p08: { coords: ["(cos(phi_120) * radius_outer)+center", "(sin(phi_120) * radius_outer)+center"], char_length: 0.3 }
          p09: { coords: ["(cos(phi_180) * radius_outer)+center", "(sin(phi_180) * radius_outer)+center"], char_length: 0.3 }
          p10: { coords: ["(cos(phi_180) * radius_inner)+center", "(sin(phi_180) * radius_inner)+center"], char_length: 0.3 }

          p11: { coords: ["(cos(phi_240) * radius_inner)+center", "(sin(phi_240) * radius_inner)+center"], char_length: 0.3 }
          p12: { coords: ["(cos(phi_240) * radius_outer)+center", "(sin(phi_240) * radius_outer)+center"], char_length: 0.3 }
          p13: { coords: ["(cos(phi_300) * radius_outer)+center", "(sin(phi_300) * radius_outer)+center"], char_length: 0.3 }
          p14: { coords: ["(cos(phi_300) * radius_inner)+center", "(sin(phi_300) * radius_inner)+center"], char_length: 0.3 }


        lines:
          g00: { start: b00, end: b01}
          g01: { start: b01, end: b02}
          g02: { start: b02, end: b03}
          g03: { start: b03, end: b00}

          a00: { start: p01, end: p02, center: p00 }
          a01: { start: p02, end: p01, center: p00 }

          l01: { start: p03, end: p04}
          a02: { start: p04, end: p05, center: p00 }
          l02: { start: p05, end: p06}
          a03: { start: p06, end: p03, center: p00 }

          l03: { start: p07, end: p08}
          a04: { start: p08, end: p09, center: p00 }
          l04: { start: p09, end: p10}
          a05: { start: p10, end: p07, center: p00 }

          l05: { start: p11, end: p12}
          a06: { start: p12, end: p13, center: p00 }
          l06: { start: p13, end: p14}
          a07: { start: p14, end: p11, center: p00 }

        surfaces:

          s1:
            loops:
              - [a00, a01]
          s2:
            loops:
              - [l01, a02, l02, a03]
          s3:
            loops:
              - [l03, a04, l04, a05]
          s4:
            loops:
              - [l05, a06, l06, a07]

          box:
            loops:
              - [g00, g01, g02, g03]
              - ["-a00", "-a01"]
              - ["-l01", "-a02", "-l02", "-a03"]
              - ["-l03", "-a04", "-l04", "-a05"]
              - ["-l05", "-a06", "-l06", "-a07"]

        groups:
            Background:
              dim: 2
              entities: [box]
            Foreground:
              dim: 2
              entities: [s1, s2, s3, s4]
            """
        )

        yaml_file = tmp_path / "geom.yml"
        yaml_file.write_text(yaml_content)

        msh_file = meshes_from_yaml(yaml_file, tmp_path)
        print(f"mesh-file: {msh_file}")

        assert msh_file.exists()

        gmsh.initialize()
        gmsh.open(str(msh_file))

        groups = gmsh.model.getPhysicalGroups()
        names = [
            gmsh.model.getPhysicalName(dim, tag).lower() for dim, tag in groups
        ]

        gmsh.finalize()

        assert "background" in names
        assert "foreground" in names
        assert len(groups) == 2

        # Check if convertible to vtk
        meshes = ot.meshes_from_gmsh(msh_file, reindex=True, log=False)
        assert meshes, "No meshes returned"

        assert "domain" in meshes, "No 'domain' mesh found"
        assert "Background" in meshes, "No 'Background' mesh found"
        assert "Foreground" in meshes, "No 'Foreground' mesh found"

        domain = meshes["domain"]
        assert 2000 < domain.n_points < 4000, "Wrong number of points"
        assert 4000 < domain.n_cells < 8000, "Wrong number of cells"

    def test_mfy_hlw_repository(self, tmp_path):
        # Load YAML geometry definition directly from file

        msh_file = meshes_from_yaml(examples.example_hlw, tmp_path)
        assert msh_file.exists()
        print(f"mesh-file: {msh_file}")

        # Collect physical group names as returned by gmsh
        gmsh.initialize()
        gmsh.open(str(msh_file))
        groups = gmsh.model.getPhysicalGroups()
        names = [gmsh.model.getPhysicalName(dim, tag) for dim, tag in groups]
        gmsh.finalize()

        # Expected names
        expected = {
            "Observation_Point",
            "Domain_Left",
            "Domain_Top",
            "Domain_Right",
            "Domain_Bottom",
            "Hostrock",
            "EDZ",
            "Support",
            "Floor",
            "Canister",
            "Filling",
        }

        missing = expected - set(names)
        unexpected = set(names) - expected
        assert not missing, f"Missing expected groups: {missing}"
        assert not unexpected, f"Unexpected extra groups: {unexpected}"

        # Check if convertible to vtk
        meshes = ot.meshes_from_gmsh(msh_file, reindex=True, log=False)
        assert meshes, "No meshes returned"

        # Build expected mesh keys: "domain" + prefixed group names
        expected_meshes = {"domain"} | {f"{name}" for name in expected}

        missing_meshes = expected_meshes - set(meshes.keys())
        unexpected_meshes = set(meshes.keys()) - expected_meshes
        assert not missing_meshes, f"Missing expected meshes: {missing_meshes}"
        assert (
            not unexpected_meshes
        ), f"Unexpected extra meshes: {unexpected_meshes}"

        domain = meshes["domain"]
        assert 3000 < domain.n_points < 4000, "Wrong number of points"
        assert 5000 < domain.n_cells < 7000, "Wrong number of cells"

    def test_mfy_hlw_repository_meshes_container(self):
        meshes = ot.Meshes.from_yaml(examples.example_hlw)

        # Check if the keys match
        expected = {
            "Observation_Point",
            "Domain_Left",
            "Domain_Top",
            "Domain_Right",
            "Domain_Bottom",
            "Hostrock",
            "EDZ",
            "Support",
            "Floor",
            "Canister",
            "Filling",
        }

        expected_meshes = {"domain"} | {f"{name}" for name in expected}

        assert set(meshes.keys()) == expected_meshes, (
            f"Unexpected mesh keys: {set(meshes.keys()) - expected_meshes} "
            f"Missing: {expected_meshes - set(meshes.keys())}"
        )

        # Check domain
        domain = meshes.domain()
        assert domain.n_points > 2000
        assert domain.n_cells > 4000

        # Test access to subdomains
        subdomains = meshes.subdomains()
        assert all(isinstance(m, ot.Mesh) for m in subdomains.values())
        assert "Floor" in subdomains
        assert "Canister" in subdomains

        # Test saving (writes temporary VTUs)
        files = meshes.save()
        assert all(f.exists() for f in files)
