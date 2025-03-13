"""Unit tests for meshlib."""

from pathlib import Path
from tempfile import mkdtemp

import matplotlib.pyplot as plt
import numpy as np
import pytest
import pyvista as pv
from lxml import etree as ET

import ogstools as ot
from ogstools import examples


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
    def test_read_quadratic_xdmf(self):
        "Test reading quadratic xdmf meshes. Tests the special case with a mesh with only 1 cell. Doesn't work with native meshio."
        tmp_dir = Path(mkdtemp())
        mesh_path = tmp_dir / "mesh.msh"
        for quads in [True, False]:
            ot.meshlib.rect(
                1, 1, structured_grid=quads, order=2, out_name=mesh_path
            )
            meshes = ot.meshes_from_gmsh(mesh_path, log=False)
            for name, mesh in meshes.items():
                pv.save_meshio(Path(tmp_dir, name + ".vtu"), mesh)

            model = ot.Project(
                output_file=tmp_dir / "default.prj",
                input_file=examples.prj_mechanics,
            )
            model.replace_text("XDMF", xpath="./time_loop/output/type")
            model.replace_text(4, xpath=".//integration_order")
            model.write_input()
            model.run_model(write_logs=False, args=f"-m {tmp_dir} -o {tmp_dir}")
            ot.MeshSeries(tmp_dir / "mesh_domain.xdmf").mesh(0)

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

    def test_plot_probe(self):
        """Test creation of probe plots."""
        meshseries = examples.load_meshseries_THM_2D_PVD()
        points = meshseries.mesh(0).center
        _ = meshseries.plot_probe(points, ot.variables.temperature)
        points = meshseries.mesh(0).points[[0, -1]]
        _ = meshseries.plot_probe(points, ot.variables.temperature)
        _ = meshseries.plot_probe(points, ot.variables.velocity)
        _ = meshseries.plot_probe(points, ot.variables.stress)
        _ = meshseries.plot_probe(points, ot.variables.stress.von_Mises)
        mesh_series = examples.load_meshseries_HT_2D_XDMF()
        points = mesh_series.mesh(0).center
        _ = meshseries.plot_probe(points, ot.variables.temperature)
        _ = meshseries.plot_probe(points, ot.variables.velocity)
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

    def test_depth_2D(self):
        mesh = examples.load_mesh_mechanics_2D()
        mesh["depth"] = mesh.depth(use_coords=True)
        # y Axis is vertical axis
        assert np.all(mesh["depth"] == -mesh.points[..., 1])
        mesh["depth"] = mesh.depth()
        assert np.all(mesh["depth"] < -mesh.points[..., 1])

    def test_depth_3D(self):
        mesh = ot.Mesh(pv.SolidSphere(100, center=(0, 0, -101)))
        mesh["depth"] = mesh.depth(use_coords=True)
        assert np.all(mesh["depth"] == -mesh.points[..., -1])
        mesh["depth"] = mesh.depth()
        assert np.all(mesh["depth"] < -mesh.points[..., -1])

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
    def test_ip_mesh(self, elem_order, quads, intpt_order, mixed):
        "Test creation of integration point meshes."

        tmp_path = Path(mkdtemp())
        mesh_path = Path(tmp_path) / "mesh.msh"
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
        meshes = ot.meshes_from_gmsh(mesh_path, log=False)
        for name, mesh in meshes.items():
            pv.save_meshio(Path(tmp_path, name + ".vtu"), mesh)
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
        assert isinstance(ot.Mesh.read(examples.mechanics_vtu), ot.Mesh)

    @pytest.mark.system()
    def test_xdmf_quadratic(self):
        "Test reading of quadratic elements in xdmf."

        tmp_path = Path(mkdtemp())
        msh_path = Path(tmp_path) / "mesh.msh"
        ot.meshlib.rect(
            n_edge_cells=6, structured_grid=False, order=2, out_name=msh_path
        )
        meshes = ot.meshes_from_gmsh(msh_path, log=False)
        for name, mesh in meshes.items():
            pv.save_meshio(Path(tmp_path, name + ".vtu"), mesh)
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

    def test_remesh_with_tri(self):
        mesh = examples.load_meshseries_THM_2D_PVD().mesh(1)
        temp_dir = Path(mkdtemp())
        msh_path = temp_dir / "tri_mesh.msh"
        ot.meshlib.gmsh_meshing.remesh_with_triangles(mesh, msh_path)
        assert len(
            ot.meshes_from_gmsh(msh_path, reindex=False, log=False).items()
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

    def test_save_pvd_mesh_series(self):
        temp = Path(mkdtemp())
        file_name = "test.pvd"

        ms = examples.load_meshseries_HT_2D_PVD()
        ms.save(Path(temp, file_name), deep=True)
        ms_test = ot.MeshSeries(Path(temp, file_name))
        assert len(ms.timevalues) == len(ms_test.timevalues)
        assert np.abs(ms.timevalues[1] - ms_test.timevalues[1]) < 1e-14
        for var in ["temperature", "darcy_velocity", "pressure"]:
            val_ref = np.sum(ms.aggregate_over_domain(var, np.max))
            val_test = np.sum(ms_test.aggregate_over_domain(var, np.max))
            assert np.abs(val_ref - val_test) < 1e-14

        for m in ms_test:
            assert "test" in m.filepath.name

        # Smoke test for ascii output
        ms.save(Path(temp, "test_ascii.pvd"), ascii=True)

        ms.save(Path(temp, file_name), deep=False)
        tree = ET.parse(Path(temp, file_name))
        num_slices = len(ms.timevalues)
        num_slices_test = len(tree.findall("./Collection/DataSet"))
        assert num_slices == num_slices_test
        pvd_entries = tree.findall("./Collection/DataSet")
        for i in range(num_slices):
            assert ms[i].filepath.name == pvd_entries[i].attrib["file"]
            ts = float(pvd_entries[i].attrib["timestep"])
            assert np.abs(ms.timevalues[i] - ts) < 1e-14

    def test_save_xdmf_mesh_series(self):
        temp = Path(mkdtemp())
        file_name = "test.pvd"

        ms = examples.load_meshseries_CT_2D_XDMF()
        ms.save(Path(temp, file_name), deep=True)
        ms_test = ot.MeshSeries(Path(temp, file_name))
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

        ms.save(Path(temp, file_name), deep=False)
        tree = ET.parse(Path(temp, file_name))
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

    def test_extend(self):
        ms1 = examples.load_meshseries_HT_2D_PVD()
        len_orig_ms1 = len(ms1)
        ms1_orig_tv = ms1.timevalues
        ms1.extend(ms1)
        print(ms1)  # check if representable
        assert len(ms1) == 2 * len_orig_ms1 - 1
        assert ms1.timevalues[-1] == ms1_orig_tv[-1] * 2
        ms1 = examples.load_meshseries_HT_2D_PVD()
        ms2 = examples.load_meshseries_HT_2D_PVD()
        ms2._timevalues = ms1.timevalues[-1] + ms2.timevalues
        ms1.extend(ms2)
        assert len(ms1) == 2 * len_orig_ms1 - 1
        assert ms1.timevalues[-1] == ms1_orig_tv[-1] * 2
        ms1 = examples.load_meshseries_HT_2D_PVD()
        ms2 = examples.load_meshseries_HT_2D_PVD()
        ms2._timevalues = ms1_orig_tv[-1] + ms1_orig_tv + 0.1
        ms1.extend(ms2)
        assert len(ms1) == 2 * len_orig_ms1
        assert ms1.timevalues[-1] == ms1_orig_tv[-1] * 2 + 0.1
