"""Unit tests for meshlib."""

import numpy as np
import pytest
import pyvista as pv
from hypothesis import given, settings
from hypothesis import strategies as st
from lxml import etree as ET

import ogstools as ot
from ogstools import examples


def test_meshseries_xdmf():
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
@pytest.mark.parametrize("quads", [True, False], ids=["quads", "no quads"])
def test_read_quadratic_xdmf(tmp_path, quads):
    "Test reading quadratic xdmf meshes. Tests the special case with a mesh with only 1 cell. Doesn't work with native meshio."
    meshes = ot.Meshes.from_gmsh(
        ot.gmsh_tools.rect(1, 1, structured_grid=quads, order=2)
    )

    prj1 = ot.Project(input_file=examples.prj_mechanics).copy("prj1.prj")
    prj1.replace_text("XDMF", xpath="./time_loop/output/type")
    prj1.replace_text(4, xpath=".//integration_order")

    model = ot.Model(prj1, meshes)
    # private function call allowed because it is guaranteed the path is empty
    model._next_target = tmp_path  # use only in testing!
    sim1 = model.run("new1", overwrite=True)
    print(sim1.log_file)
    ms_domain = sim1.meshseries
    if quads:
        # 4 corners, 4 between corners, 1 center
        assert ms_domain[-1].number_of_points == 8
    else:  # triangles
        # 4 corners, 4 between corners, 1 center, 4 between corner and center
        assert ms_domain[-1].number_of_points == 13


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
def test_meshseries_fileformats_indexing(ht):
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
def test_meshseries_values(ht):
    assert np.shape(ht.values("temperature")) == (1, 190)
    assert np.shape(ht.values("darcy_velocity")) == (1, 190, 2)
    assert np.shape(ht.values("temperature")[:, 0:5]) == (1, 5)
    assert np.shape(ht.values("darcy_velocity")[:, 1:4, :]) == (1, 3, 2)


def test_meshseries_xdmf_rawdata():
    xdmf = examples.load_meshseries_HT_2D_XDMF()
    h5file = xdmf.rawdata_file()
    assert h5file is not None
    assert h5file.suffix == ".h5"

    # This XDMF file is not generate via OGS/OGSTools, therefore the
    # underlying file structure is not known and no optimization is possible.
    xmf = examples.load_meshseries_HT_2D_paraview_XMF()
    assert xmf.rawdata_file() is None


def test_all_types():
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

        assert ms.timevalues[ms.closest_timestep(1.0)] == ms.closest_timevalue(
            1.0
        )

        ms.clear_cache()


def test_items():
    ms = examples.load_meshseries_HT_2D_XDMF()
    for i, (timevalue, mesh) in enumerate(ms.items()):
        assert timevalue == ms.timevalues[i]
        assert mesh == ms[i]


def test_cache():
    ms = examples.load_meshseries_HT_2D_PVD()
    assert not ms._is_all_cached  # pylint: disable=W0212
    _ = ms[::2]
    assert not ms._is_all_cached  # pylint: disable=W0212
    _ = ms.values("temperature")
    assert ms._is_all_cached  # pylint: disable=W0212


def test_datadict():
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

    for ms_key, mesh_key in zip(ms.point_data, ms[0].point_data, strict=True):
        assert ms_key == mesh_key
    assert len(ms.cell_data) == len(ms[0].cell_data)


def test_cache_copy():
    "Test that the cache of a MeshSeries is a deep copy as well."
    ms = examples.load_meshseries_HT_2D_XDMF()
    _ = ms[0]
    ms_subset = ms.copy().extract(0, "cells")
    assert ms[0].number_of_cells > ms_subset[0].number_of_cells


def test_reindexing():
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


def test_temporal_aggregate():
    "Test aggregation of meshseries."
    mesh_series = examples.load_meshseries_HT_2D_XDMF()
    funcs = [np.min, np.max, np.mean, np.median, np.sum, np.std, np.var]
    for func in funcs:
        agg_mesh = mesh_series.aggregate_temporal("temperature", func)
        assert not np.any(np.isnan(agg_mesh["temperature_" + func.__name__]))


def test_aggregate_time_of_limit():
    "Test aggregation of meshseries."
    mesh_series = examples.load_meshseries_HT_2D_XDMF()
    min_mesh = mesh_series.time_of_min("temperature")
    assert not np.any(np.isnan(min_mesh["min_temperature_time"]))
    max_mesh = mesh_series.time_of_max("temperature")
    assert not np.any(np.isnan(max_mesh["max_temperature_time"]))


def test_temporal_aggregate_mesh_dependent():
    "Test aggregation of mesh_dependent variable on meshseries."
    mesh_series = examples.load_meshseries_THM_2D_PVD()
    prop = ot.variables.dilatancy_alkan
    agg_mesh = mesh_series.aggregate_temporal(prop, np.max)
    assert not np.any(np.isnan(agg_mesh[prop.output_name + "_max"]))
    agg_mesh = mesh_series.time_of_max(prop)
    assert not np.any(np.isnan(agg_mesh[f"max_{prop.output_name}_time"]))


def test_spatial_aggregate():
    "Test aggregation of meshseries."
    ms = examples.load_meshseries_HT_2D_XDMF()[1:]
    temp_min = ms.aggregate_spatial("temperature", np.min)
    temp_mean = ms.aggregate_spatial("temperature", np.mean)
    temp_max = ms.aggregate_spatial("temperature", np.max)
    assert np.all(temp_max > temp_mean)
    assert np.all(temp_mean > temp_min)


def _check_probe(ms: ot.MeshSeries, points: np.ndarray) -> None:
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
            values = ms.probe_values(points[slicing], "temperature", method)
            assert values.shape == shape
            np.testing.assert_allclose(values, ref_values[:, slicing])
            assert not np.any(np.isnan(values))


def test_probe_2D_mesh():
    "Test point probing on a 2D meshseries."
    ms = examples.load_meshseries_HT_2D_XDMF()
    pt_min, pt_max = np.reshape(ms[0].bounds, (3, 2)).T
    points = np.linspace(pt_min, pt_max, num=10)
    _check_probe(ms, points)


def test_probe_1D_mesh():
    "Test point probing on a 1D meshseries."
    ms = examples.load_meshseries_HT_2D_XDMF()
    ms_1D = ms.extract(ms[0].points[:, 1] == 0)
    pt_min, pt_max = np.reshape(ms_1D[0].bounds, (3, 2)).T
    points = np.linspace(pt_min, pt_max, num=10)
    _check_probe(ms_1D, points)


def test_probe_1D_mesh_single_pt():
    "Test single point probing on a 1D meshseries."
    ms = examples.load_meshseries_HT_2D_XDMF()
    ms_1D = ms.extract(ms[0].points[:, 1] == 0)
    pt_min = np.reshape(ms_1D[0].bounds, (3, 2)).T[0]
    ms_ref = ms_1D.probe(pt_min)
    for key in ms_ref.point_data:
        np.testing.assert_array_equal(
            ms_ref.values(key)[:, 0], ms_1D.values(key)[:, 0]
        )


def test_probe_multiple():
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
        ms_pts = ms.probe(points, arg)
        for key in keys:
            ms_ref = ms.probe(points, key)
            np.testing.assert_array_equal(
                ms_pts.values(key), ms_ref.values(key)
            )


def test_probe():
    results = examples.load_meshseries_HT_2D_XDMF()
    points = np.linspace([2, 2, 0], [4, 18, 0], num=100)
    ms_pts = results.probe(points, "temperature")
    np.testing.assert_array_equal(ms_pts[0].points, points)
    pts_mesh = pv.PolyData(points)
    ms_pts = results.probe(pts_mesh, "temperature", "nearest")
    pt_id = results[0].find_closest_point(points[0])
    np.testing.assert_array_equal(
        ms_pts["temperature"][:, 0], results["temperature"][:, pt_id]
    )


def test_temporal_resample():
    results = examples.load_meshseries_HT_2D_XDMF()
    in_between = 0.5 * (results.timevalues[:-1] + results.timevalues[1:])
    resampled = results.resample_temporal(in_between)
    for idx, mesh in enumerate(resampled):
        for var in ["temperature", "pressure", "darcy_velocity"]:
            delta = results[idx + 1][var] - results[idx][var]
            half_delta = mesh[var] - results[idx][var]
            np.testing.assert_almost_equal(half_delta, 0.5 * delta)


def test_scaling():
    ms = examples.load_meshseries_THM_2D_PVD()
    bounds_m = ms[0].bounds
    ms.scale("km")
    np.testing.assert_allclose(bounds_m, 1e3 * np.asarray(ms[0].bounds))
    ms.scale(1e3)
    np.testing.assert_allclose(bounds_m, ms[0].bounds)

    tv_s = ms.timevalues
    ms.scale(time="d")
    np.testing.assert_allclose(tv_s, 86400 * ms.timevalues)
    ms.scale(time=86400)
    np.testing.assert_allclose(tv_s, ms.timevalues)


@pytest.mark.parametrize(
    ("ms", "var"),
    [
        (examples.load_meshseries_THM_2D_PVD(), ot.variables.temperature),
        (examples.load_meshseries_THM_2D_PVD(), ot.variables.pressure),
        (examples.load_meshseries_CT_2D_XDMF(), ot.variables.saturation),
        (
            examples.load_meshseries_HT_2D_XDMF(),
            ot.variables.velocity.replace(data_name="darcy_velocity"),
        ),
        (
            examples.load_meshseries_BHEs_3D("full", ".xdmf"),
            ot.variables.temperature_BHE[1, 1],
        ),
    ],
)
def test_diff_meshseries(ms, var: ot.variables.Variable):
    """
    Tests using difference method:
        A) By computing difference with self -> zero.
        B) By computing difference with doubled self ->
    """
    variable = var.replace(output_unit=var.data_unit)  # needed for double check

    # A) Diff with self, should result in only 0 values
    ms_diff = ot.MeshSeries.difference(ms, ms, variable)
    assert isinstance(ms_diff, ot.MeshSeries)
    assert len(ms) == len(ms_diff)
    for mesh in ms_diff:
        # np.nan_to_num is needed to account for mask in pressure in THM_2D
        data = np.nan_to_num(mesh[variable.difference.output_name])
        assert np.count_nonzero(data) == 0

    # B) Diff with self*2 should result in values equal to self
    ms_double = ms.copy()
    for m in ms_double:
        m[variable.data_name] = m[variable.data_name] * 2
    ms_zero = ot.MeshSeries.difference(ms_double, ms, variable)
    assert isinstance(ms_zero, ot.MeshSeries)
    assert len(ms) == len(ms_zero)
    for mesh_diff, mesh in zip(ms_zero, ms, strict=True):
        assert np.array_equal(
            mesh_diff[variable.difference.data_name],
            # Below: mesh has to be transformed to keep unit conversion consistent
            # and for temperature_BHE to select one column from a vector
            variable.transform(mesh),
            equal_nan=True,  # for cases with masks (subdomain deactivated)
        )


@pytest.mark.parametrize(
    ("ms"),
    [
        (examples.load_meshseries_THM_2D_PVD()),
        (examples.load_meshseries_HT_2D_XDMF()),
        (examples.load_meshseries_BHEs_3D("full", ".xdmf")),
    ],
)
def test_raw_diff_meshseries(ms):
    ms = examples.load_meshseries_THM_2D_PVD()

    # Diff with self, should result in only 0 values
    ms_diff = ot.MeshSeries.difference(ms, ms)
    assert isinstance(ms_diff, ot.MeshSeries)
    assert len(ms) == len(ms_diff)

    for key in ms_diff.point_data:
        assert np.count_nonzero(ms_diff[key]) == 0

    for key in ms_diff.cell_data:
        if key == "MaterialIDs":
            continue
        assert np.count_nonzero(ms_diff[key]) == 0


@pytest.mark.parametrize(
    ("ms", "var"),
    [
        (examples.load_meshseries_THM_2D_PVD(), ot.variables.temperature),
        (examples.load_meshseries_THM_2D_PVD(), ot.variables.pressure),
        (examples.load_meshseries_CT_2D_XDMF(), ot.variables.saturation),
        (
            examples.load_meshseries_BHEs_3D("full", ".xdmf"),
            ot.variables.temperature_BHE[1, 1],
        ),
    ],
)
def test_compare_meshseries(ms, var):
    """
    This test attempts to check `MeshSeries.compare` s.t.:
        (A) Identical MeshSeries: compare(ms, ms)==True
        (B) compare MeshSeries with different topologies
        (C) compare MeshSeries with different timevalues
    """
    # (A) compare with itself
    assert ot.MeshSeries.compare(ms, ms, var)
    assert ot.MeshSeries.compare(ms, ms)

    # (B) Test different topology
    ms_scaled = ms.copy()
    ms_scaled.scale(2.0)
    assert not ot.MeshSeries.compare(ms_scaled, ms, var)
    with pytest.raises(
        AssertionError,
        match="The topologies of the MeshSeries objects are not identical.",
    ):
        ot.MeshSeries.compare(ms_scaled, ms, var, strict=True)

    # (C) Test different timevalues
    ms_time_shifted = ms.copy()
    ms_time_shifted._timevalues += 1
    assert not ot.MeshSeries.compare(ms, ms_time_shifted, var)
    with pytest.raises(
        AssertionError,
        match="timevalues differs between MeshSeries.",
    ):
        ot.MeshSeries.compare(ms_time_shifted, ms, var, strict=True)
    return


@given(
    tols=st.builds(
        lambda atol, frac_in, sign, mult: (
            atol,
            frac_in * atol,
            sign * mult * atol,
        ),
        atol=st.floats(min_value=1e-12, max_value=2.0),
        frac_in=st.floats(-0.99, 0.99).filter(lambda x: x != 0),
        sign=st.sampled_from([-1.0, 1.0]),
        mult=st.floats(min_value=1.01, max_value=10.0),
    )
)
@settings(deadline=500)
def test_compare_meshseries_tol(tols):
    """
    This test attempts to check `MeshSeries.compare` atol using property-testing

    :param tols:    `tols` is a tuple generated by `hypothesis` s.t. `tols = (atol, tol_in, tol_out)` where:
                        - `atol` is drawn randomly using `hypothesis`
                        - `tol_in = frac_in * atol` w/ `frac_in \\in [-0.99, 0.99] \\ {0}`
                        - `tol_out = sign * mult * atol` w/ `sign \\in {-1, 1}` and `mult \\in [1.01, 10.0]`
    """
    ms = examples.load_meshseries_THM_2D_PVD()
    var = ot.variables.temperature
    atol, tol_in, tol_out = tols

    # (A) Scaling var data: compare ms with ms + tol_in
    ms_a = ms.copy()
    for m in ms_a:
        m[var.data_name] += tol_in
    assert ot.MeshSeries.compare(ms, ms_a, var, atol=atol)

    # (B) Scaling var data: compare ms with ms + tol_out
    ms_b = ms.copy()
    for m in ms_b:
        m[var.data_name] += tol_out
    assert not ot.MeshSeries.compare(ms, ms_b, var, atol=atol)
    with pytest.raises(
        AssertionError, match=f"{var.data_name} differs between MeshSeries."
    ):
        ot.MeshSeries.compare(ms, ms_b, atol=atol, strict=True)
    return


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
def test_ip_mesh(tmp_path, elem_order, quads, intpt_order, mixed):
    "Test creation of integration point meshes."
    sigma_ip = ot.variables.stress.replace(data_name="sigma_ip")

    rect = ot.gmsh_tools.rect(
        n_edge_cells=6,
        n_layers=2,
        structured_grid=quads,
        order=elem_order,
        mixed_elements=mixed,
        jiggle=0.01,
    )
    meshes = ot.Meshes.from_gmsh(rect)
    prj = ot.Project(input_file=examples.prj_mechanics).copy()
    prj.replace_text(intpt_order, xpath=".//integration_order")

    # ToDo log prj.run_model(write_logs=True, args=f"-m {tmp_path} -o {tmp_path}")
    model = ot.Model(prj, meshes)
    model._next_target = tmp_path  # use only in testing!
    meshseries = model.run().meshseries
    mesh_last = meshseries[-1]
    int_pts = ot.mesh.to_ip_point_cloud(mesh_last)

    ip_ms = meshseries.ip_tesselated()
    ip_mesh = ip_ms.mesh(-1)
    vals = ip_ms.probe_values(ip_mesh.center, sigma_ip.data_name)
    assert not np.any(np.isnan(vals))
    assert int_pts.number_of_points == ip_mesh.number_of_cells
    containing_cells = ip_mesh.find_containing_cell(int_pts.points)
    # check for integration points coinciding with the tessellated cells
    np.testing.assert_equal(
        sigma_ip.magnitude.transform(ip_mesh)[containing_cells],
        sigma_ip.magnitude.transform(int_pts),
    )


def test_reader():
    assert isinstance(examples.load_meshseries_THM_2D_PVD(), ot.MeshSeries)
    assert isinstance(ot.MeshSeries(examples.elder_xdmf), ot.MeshSeries)


@pytest.mark.system()
def test_xdmf_quadratic(tmp_path):
    "Test reading of quadratic elements in xdmf."

    meshes = ot.Meshes.from_gmsh(
        ot.gmsh_tools.rect(n_edge_cells=6, structured_grid=False, order=2)
    )
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


def test_indexing():
    ms = examples.load_meshseries_HT_2D_XDMF()
    assert isinstance(ms[1], pv.UnstructuredGrid)


def test_slice():
    ms = examples.load_meshseries_HT_2D_XDMF()
    ms_sliced = ms[1::2]
    assert len(ms.timevalues) >= 2 * len(ms_sliced.timevalues)


def test_transform():
    ms = examples.load_meshseries_THM_2D_PVD()
    ms_mod = ms.transform(lambda mesh: mesh.slice("x"))
    assert max(ms[0].cells) == 3779  # Check if example mesh has changed
    assert max(ms_mod[0].cells) == 44
    assert len(ms[0].points) == 3780  # Check if example mesh has changed
    assert len(ms_mod[0].points) == 45


def test_copy_deep():
    ms = examples.load_meshseries_THM_2D_PVD()
    ms.point_data["temperature"] = 0
    ms_deepcopy = ms.copy(deep=True)
    ms_deepcopy.point_data["temperature"] = 1
    assert np.all(
        ms.point_data["temperature"] != ms_deepcopy.point_data["temperature"]
    )


def test_copy_shallow():
    ms = examples.load_meshseries_THM_2D_PVD()
    ms.point_data["temperature"] = 0
    ms_shallowcopy = ms.copy(deep=False)
    ms_shallowcopy.point_data["temperature"] = 1
    assert np.all(
        ms.point_data["temperature"] == ms_shallowcopy.point_data["temperature"]
    )


def test_save_pvd_mesh_series(tmp_path):
    file_name = "test.pvd"

    ms = examples.load_meshseries_HT_2D_PVD()
    ms.save(tmp_path / file_name, deep=True)
    ms_test = ot.MeshSeries(tmp_path / file_name)
    assert len(ms.timevalues) == len(ms_test.timevalues)
    assert np.abs(ms.timevalues[1] - ms_test.timevalues[1]) < 1e-14
    for var in ["temperature", "darcy_velocity", "pressure"]:
        val_ref = np.sum(ms.aggregate_spatial(var, np.max))
        val_test = np.sum(ms_test.aggregate_spatial(var, np.max))
        assert np.abs(val_ref - val_test) < 1e-14

    for m in ms_test:
        assert "test" in m.filepath.name

    # Smoke test for ascii output
    ms.save(tmp_path / "test_ascii.pvd", ascii=True)

    ms.save(tmp_path / file_name, deep=False, overwrite=True)
    tree = ET.parse(tmp_path / file_name)
    num_slices = len(ms.timevalues)
    num_slices_test = len(tree.findall("./Collection/DataSet"))
    assert num_slices == num_slices_test
    pvd_entries = tree.findall("./Collection/DataSet")
    for i in range(num_slices):
        assert ms[i].filepath.name == pvd_entries[i].attrib["file"]
        ts = float(pvd_entries[i].attrib["timestep"])
        assert np.abs(ms.timevalues[i] - ts) < 1e-14


def test_save_xdmf_mesh_series(tmp_path):
    file_name = "test.pvd"

    ms = examples.load_meshseries_CT_2D_XDMF()
    ms.save(tmp_path / file_name, deep=True)
    ms_test = ot.MeshSeries(tmp_path / file_name)
    assert len(ms.timevalues) == len(ms_test.timevalues)
    assert np.abs(ms.timevalues[1] - ms_test.timevalues[1]) < 1e-14
    assert (
        np.abs(
            np.sum(ms.aggregate_spatial("Si", np.max))
            - np.sum(ms_test.aggregate_spatial("Si", np.max))
        )
        < 1e-14
    )
    for m in ms_test:
        assert "test" in m.filepath.name

    file_name = "test_shallow.pvd"
    ms.save(tmp_path / file_name, deep=False)
    tree = ET.parse(tmp_path / file_name)
    num_slices = len(ms.timevalues)
    pvd_entries = tree.findall("./Collection/DataSet")
    num_slices_test = len(pvd_entries)
    assert num_slices == num_slices_test
    for i in range(num_slices):
        ts = float(pvd_entries[i].attrib["timestep"])
        assert np.abs(ms.timevalues[i] - ts) < 1e-14


def test_remove_array():

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
def test_extend(load_example_ms):
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


def test_ms_active():
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
    m_diff_T = ot.mesh.difference(ms[-1], ms[0], ot.variables.temperature)
    m_diff_p = ot.mesh.difference(ms[-1], ms[0], ot.variables.pressure)
    assert (
        np.count_nonzero(
            np.isnan(m_diff_T[ot.variables.temperature.difference.data_name])
        )
        == 0
    )
    assert (
        np.count_nonzero(
            np.isnan(m_diff_p[ot.variables.pressure.difference.data_name])
        )
        != 0
    )
