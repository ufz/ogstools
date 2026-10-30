"""Unit tests for variables."""

from collections.abc import Callable
from itertools import pairwise

import numpy as np
import pytest
from pint.facets.plain import PlainQuantity

import ogstools as ot
from ogstools import examples
from ogstools import variables as ov
from ogstools.mesh.utils import angles

Qty = ov.u_reg.Quantity

bhe_mesh_series = {
    "1P": examples.load_meshseries_BHE_3D_1P(),
    "1U": examples.load_meshseries_BHE_3D_1U(),
    "2U": examples.load_meshseries_BHE_3D_2U(),
    "CXA": examples.load_meshseries_BHE_3D_CXA(),
    "CXC": examples.load_meshseries_BHE_3D_CXC(),
}


class TestPhysicalVariable:
    """Test case for physical variables."""

    EPS = 1e-7

    def equality(self, v: ov.Scalar, vals: np.ndarray, res: PlainQuantity):
        """
        Assert the equality of variable calculations.

        :param v: The scalar variable to test.
        :param vals: The input values.
        :param res: The expected result.
        """
        np.testing.assert_allclose(
            v.transform(vals),
            res.to(v.output_unit).magnitude,
            rtol=self.EPS,
            verbose=True,
        )

    def test_temperature(self):
        """Test temperature variable."""

        self.equality(ov.temperature, 293.15, Qty(20, "°C"))
        self.equality(ov.temperature, [293.15, 303.15], Qty([20, 30], "°C"))

    def test_pressure(self):
        """Test pressure variable."""
        self.equality(ov.pressure, 1e6, Qty(1, "MPa"))
        self.equality(ov.pressure, [1e6, 2e6], Qty([1, 2], "MPa"))

    def test_velocity(self):
        """Test velocity variable."""
        self.equality(ov.velocity.magnitude, [3, 4], Qty(5, "m/s"))
        self.equality(
            ov.velocity.magnitude, [[3, 4], [1, 0]], Qty([5, 1], "m/s")
        )

    def test_displacement(self):
        """Test displacement variable."""
        self.equality(ov.displacement.magnitude, [3e-3, 4e-3], Qty(5, "mm"))
        u = np.array([[2, 3, 6], [1, 4, 8]]) * 1e-3
        self.equality(ov.displacement.magnitude, u, Qty([7.0, 9.0], "mm"))

    def test_strain(self):
        """Test strain variable."""
        eps = np.array([1, 5, 8, 1, 0, 2]) * 1e-2
        self.equality(ov.strain.magnitude, eps, Qty(10, "%"))
        self.equality(ov.strain.magnitude, [eps, eps], Qty([10, 10], "%"))
        self.equality(ov.strain.trace, eps, Qty(14, "%"))
        self.equality(ov.strain.trace, [eps, eps], Qty([14, 14], "%"))

    def test_components(self):
        """Test strain components."""
        eps = np.array([0, 1, 2, 3, 4, 5]) * 1e-2
        u = np.array([0, 1, 2]) * 1e-3
        for i, _ in enumerate(eps):
            self.equality(ov.strain[i], eps, Qty(i, "%"))
            self.equality(ov.strain[i], [eps, eps], Qty([i, i], "%"))
        for i, _ in enumerate(u):
            self.equality(ov.displacement[i], u, Qty(i, "mm"))
            self.equality(ov.displacement[i], [u, u], Qty([i, i], "mm"))
        assert ov.strain[0].bilinear_cmap is True

    def test_von_mises(self):
        """Test von_mises_stress variable."""
        sig_3D = np.array([2, 2, 2, 1, 1, 1]) * 1e6
        self.equality(ov.stress.von_Mises, sig_3D, Qty(3, "MPa"))
        self.equality(ov.stress.von_Mises, [sig_3D, sig_3D], Qty([3, 3], "MPa"))
        sig_2D = np.array([2, 2, 2, 3**0.5]) * 1e6
        self.equality(ov.stress.von_Mises, sig_2D, Qty(3, "MPa"))

    def test_eff_pressure(self):
        """Test effective_pressure variable."""
        sig = np.array([-1, -2, -3, 1, 1, 1]) * 1e6
        self.equality(ov.effective_pressure, sig, Qty(2, "MPa"))
        self.equality(ov.effective_pressure, [sig, sig], Qty([2, 2], "MPa"))

    def test_qp_ratio(self):
        """Test qp_ratio variable."""
        sig = np.array([4, 4, 4, 1, 1, 1]) * 1e6
        self.equality(ov.stress.qp_ratio, sig, Qty(-75, "percent"))
        self.equality(ov.stress.qp_ratio, [sig] * 2, Qty([-75] * 2, "percent"))

    def test_integrity_criteria(self):
        """Test integrity criteria."""
        mesh = examples.load_mesh_mechanics_2D()
        assert np.max(ov.dilatancy_alkan.transform(mesh)) > 0
        assert np.max(ov.dilatancy_critescu.transform(mesh)) > 0
        sigma = mesh["sigma"]
        fpc = ov.fluid_pressure_criterion_clay
        fpc_vals = fpc.transform(mesh)
        # repeated evaluation must neither change the result nor the mesh
        np.testing.assert_array_equal(fpc_vals, fpc.transform(mesh))
        np.testing.assert_array_equal(mesh["sigma"], sigma)

        p = -1000 * 9.81 * mesh.points[:, 1]
        mesh.point_data["water_pressure"] = p
        fps = ov.fluid_pressure_criterion_salt
        fps_vals = fps.transform(mesh)
        assert np.max(fps_vals) > 0
        p_MPa = ov.u_reg.convert(p, "Pa", "MPa")
        np.testing.assert_allclose(fps_vals, fpc_vals + p_MPa, 1e-12)

        # masked tensors give NaN instead of failing in the eigen solver
        masked_mesh = mesh.copy()
        masked_mesh["host_rock"] = (mesh["MaterialIDs"] == 1).astype(int)
        fps_vals_masked = fps.transform(masked_mesh)
        is_nan = np.isnan(fps_vals_masked)
        assert 0 < np.count_nonzero(is_nan) < len(fps_vals_masked)
        np.testing.assert_array_equal(
            fps_vals_masked[~is_nan], fps_vals[~is_nan]
        )
        np.testing.assert_array_equal(masked_mesh["sigma"], sigma)

    def test_stress_effective_total_roundtrip(self):
        """Check, that conversion of effective / total stress is reversible."""
        biot, chi = 0.6, 0.8
        stress = ov.stress.convert_stress(
            to="total", biot=biot, bishop=chi
        ).convert_stress(to="effective", biot=biot, bishop=chi)
        mesh = examples.load_meshseries_THM_2D_PVD()[-1]
        np.testing.assert_allclose(
            stress.transform(mesh), ov.stress.transform(mesh)
        )

    def test_stress_conversion_with_Bishops_field(self):
        """Test stress conversion with Bishop's coefficient from data field."""
        mesh = examples.load_mesh_mechanics_2D()
        mesh["pressure"] = ov.u_reg.convert(np.ones(mesh.n_points), "MPa", "Pa")
        ref = ov.stress.convert_stress(to="total").transform(mesh)
        mesh["chi"] = np.ones(mesh.n_cells)
        val = ov.stress.convert_stress(to="total", bishop="chi").transform(mesh)
        np.testing.assert_array_equal(ref, val)

    def test_stress_conversion_errors(self):
        """Test input validation of the total/effective stress conversion."""
        mesh = examples.load_mesh_mechanics_2D()
        with pytest.raises(KeyError, match=r"not present"):  # pressure missing
            ov.dilatancy_alkan.convert_stress(to="total").transform(mesh)
        mesh.point_data["pressure"] = np.ones(mesh.n_points)
        with pytest.raises(ValueError, match="Kelvin notation"):
            ov.displacement.convert_stress(to="total").transform(mesh)

    @pytest.mark.parametrize(
        "var", [ov.dilatancy_alkan, ov.fluid_pressure_criterion_clay]
    )
    def test_integrity_criteria_unit_correctness(self, var: ov.Variable):
        """Test integrity criteria."""
        mesh = examples.load_mesh_mechanics_2D()
        mesh.point_data["pressure"] = -1000 * 9.81 * mesh.points[:, 1]
        actual = var.transform(mesh)
        ref = var.replace(process_with_units=True).transform(mesh)
        np.testing.assert_allclose(actual, ref)

    def test_tensor_attributes(self):
        """Test that the access of tensor attributes works."""
        # data needs to be a 2D array
        sig = np.asarray([[4, 1, 2, 1, 1, 1]]) * 1e6
        assert np.all(ov.stress.eigenvalues.transform(sig) >= 0)
        for i in range(3):
            np.testing.assert_allclose(
                ov.stress.eigenvectors[i].magnitude.transform(sig), 1.0
            )
        assert ov.stress.det.transform(sig) > 0
        assert ov.stress.invariant_1.transform(sig) > 0
        assert ov.stress.invariant_2.transform(sig) > 0
        assert ov.stress.invariant_3.transform(sig) > 0
        assert ov.stress.tensor_mean.transform(sig) > 0
        assert ov.stress.deviator.magnitude.transform(sig) > 0
        assert ov.stress.deviator_invariant_1.transform(sig) > 0
        assert ov.stress.deviator_invariant_2.transform(sig) > 0
        assert ov.stress.deviator_invariant_3.transform(sig) > 0
        assert ov.stress.octahedral_shear.transform(sig) > 0

    def test_simple(self):
        """Test call functionality."""
        assert ov.temperature.transform(273.15, strip_unit=False) == Qty(
            0, "°C"
        )

        assert ov.displacement[0].transform([1, 2, 3], strip_unit=False) == Qty(
            1, "m"
        )

        assert ov.displacement.transform([1, 2, 3], strip_unit=False)[1] == Qty(
            2, "m"
        )

    def test_values(self):
        """Test values functionality."""
        assert ov.temperature.transform(273.15) == 0.0

    def test_units(self):
        """Test get_output_unit functionality."""
        assert ov.temperature.get_output_unit == "°C"
        assert ov.pressure.get_output_unit == "MPa"
        assert ov.strain.get_output_unit == "%"

    def test_mask(self):
        """Test mask functionality."""
        assert ov.temperature.get_mask().is_mask()

    def test_find_variable(self):
        """Test find variable function."""
        mesh = examples.load_mesh_mechanics_2D()
        mesh.point_data["scalar"] = mesh["temperature"]
        mesh.point_data["vector"] = mesh["displacement"]
        mesh.point_data["matrix"] = mesh["sigma"]
        assert ov.Variable.find("temperature", mesh) == ov.temperature
        assert ov.Variable.find("displacement", mesh) == ov.displacement
        assert ov.Variable.find("sigma", mesh) == ov.stress
        assert ov.Variable.find("scalar", mesh) == ov.Scalar("scalar")
        assert ov.Variable.find("vector", mesh) == ov.Vector("vector")
        assert ov.Variable.find("matrix", mesh) == ov.Matrix("matrix")
        with pytest.raises(KeyError):
            ov.Variable.find("test", mesh)
        # testcase with str matching output_name of the predefined Variables
        mesh = examples.load_meshseries_HT_2D_XDMF()[0]
        darcy = ov.Variable.find("darcy_velocity", mesh)
        assert darcy.data_name in mesh.point_data

    def test_find_variable_prefers_mesh_data_over_attribute(self):
        """Mesh data wins when its name collides with a mesh attribute."""
        mesh = examples.load_mesh_mechanics_2D()
        mesh.point_data["n_cells"] = np.arange(mesh.n_points)

        variable = ov.Variable.find("n_cells", mesh)

        assert all(variable.transform(mesh) == np.arange(mesh.n_points))

    def test_copy_ctor(self):
        """Test replace constructor."""

        strain_copy = ov.stress.replace(
            data_name=ov.strain.data_name,
            data_unit=ov.strain.data_unit,
            output_unit=ov.strain.output_unit,
            output_name=ov.strain.output_name,
            symbol=ov.strain.symbol,
            mask=ov.strain.mask,
            func=ov.strain.func,
        )

        assert ov.strain == strain_copy

    def test_get_label(self):
        assert ov.pressure.get_label() == "pore pressure $p$ / MPa"
        name_len = len(ov.stress.output_name) + 8  # for symbol and unit
        assert "\n" in ov.stress.get_label(name_len)
        assert "\n" not in ov.stress.get_label(name_len + 1)

    def test_BHE_temperature(self):
        with pytest.raises(TypeError):
            _ = ov.temperature_BHE.magnitude.transform(1)
        for _, components in ov.temperature_BHE.COMPONENTS.items():
            values = 273.15 + np.arange(len(components))
            for index, component in enumerate(components):
                assert index == ov.temperature_BHE[component].transform(values)

    def test_BHE_temperature_components(self):
        var = ov.temperature_BHE
        with pytest.raises(IndexError):
            _ = var[0, 0, 0].transform(1)
        mesh = examples.load_mesh_mechanics_2D()  # could be any mesh
        temp = np.full((mesh.n_points, 3), 273.15)
        mesh.point_data["temperature_BHE"] = temp
        mesh.point_data["temperature_BHE1"] = temp + np.asarray([1, -1, 0.5])
        mesh.point_data["temperature_BHE2"] = temp + np.asarray([2, -2, 1.0])
        assert np.all(var["in"].transform(mesh) == 0)
        for i in range(1, 3):
            assert np.all(var[i, "in"].transform(mesh) == i)
            assert np.all(var[i, "out"].transform(mesh) == -i)
            assert np.all(var[i, "grout"].transform(mesh) == 0.5 * i)

    @pytest.mark.parametrize("bhe_type", list(ov.temperature_BHE.COMPONENTS))
    @pytest.mark.parametrize(
        "index_gen",
        [
            lambda c: c,
            lambda c: [list(pair) for pair in pairwise(c)],
            lambda c: range(len(c)),
            lambda c: [list(pair) for pair in pairwise(range(len(c)))],
        ],
    )
    def test_BHE_temperature_component_indexing(self, bhe_type, index_gen):
        ms = bhe_mesh_series[bhe_type]
        index_combinations = index_gen(ov.temperature_BHE.COMPONENTS[bhe_type])
        temp = [
            ms.probe_values((0, 0, 0), ov.temperature_BHE[1, idx])[0]
            for idx in index_combinations
        ]
        # initial vector is +1 for every component
        assert np.all(np.diff(temp) == 1)

    @pytest.mark.parametrize("ext", [".pvd", ".xdmf"])
    @pytest.mark.parametrize("kind", ["full", "line", "lines"])
    def test_BHE_probe(self, kind, ext):
        ms = examples.load_meshseries_BHEs_3D(kind, ext)
        lines = ms[0].extract_cells(
            [cell.dimension == 1 for cell in ms[0].cell]
        )
        pts = [
            lines.extract_points(lines.points[:, 0] == x).center
            for x in np.unique(lines.points[:, 0])
        ]
        for bhe_idx, pt in enumerate(pts):
            vals = ms.probe_values(pt, ov.temperature_BHE[bhe_idx + 1, 0])
            # model symmetric: first and last BHE have the same temperature
            last_val = [21.998526268, 22.10904857, 21.998526268][bhe_idx]
            np.testing.assert_almost_equal(vals, [21.85, last_val])

    def test_shorthand_ctor(self):
        var_full = ov.Scalar(data_name="test", data_unit="unit",
                             output_unit="unit", output_name="test")  # fmt:skip
        var_short = ov.Scalar(data_name="test", data_unit="unit")
        assert var_full.output_unit == var_short.output_unit
        assert var_full.output_name == var_short.output_name

    def test_derived_variables(self):
        assert ov.temperature.difference.transform(1) == 1
        assert ov.temperature.abs_error.transform(1) == 1
        assert ov.temperature.rel_error.transform(0.01) == 1
        assert ov.temperature.anasol.transform(274.15) == 1

    def test_transform_meshseries(self):
        ms = examples.load_meshseries_THM_2D_PVD()

        def check_limits(
            variable: ov.Variable, vmin: float, vmax: float
        ) -> None:
            vals = variable.transform(ms)
            assert vmin <= np.nanmin(vals)
            assert vmax >= np.nanmax(vals)

        check_limits(ov.temperature, 7.9, 50)
        check_limits(ov.pressure, 0, 8)
        check_limits(ov.displacement.magnitude, 0, 0.3)

    def test_transform_variable_argument(self):

        mesh = examples.load_meshseries_THM_2D_PVD()[-1]
        for var in [ov.pressure, ov.stress.det]:
            var2 = type(var).from_variable(
                var,
                func=ov.Function(lambda x1, x2: x1 + x2, [var]),
                suffix="double",
            )
            np.testing.assert_allclose(
                var2.transform(mesh), 2.0 * var.transform(mesh)
            )

    def test_variable_rate(self):
        "Tests rate computation"
        ms = examples.load_meshseries_CT_2D_XDMF()
        var = ov.saturation.rate("d")
        for values in [var.mean.transform(ms), ms.values(var.mean)]:
            np.testing.assert_array_less(0.0, np.nan_to_num(values, nan=1e-12))
        assert not np.any(np.isnan(ms.probe_values((0, 0, 0), var)[1:]))

    def test_abs(self):
        "Tests rate computation"
        ms = examples.load_meshseries_CT_2D_XDMF()
        var = ov.saturation.abs
        for values in [var.transform(ms), ms.values(var)]:
            assert np.all(values >= 0.0)

    @pytest.mark.parametrize(
        "ms",
        [
            examples.load_meshseries_THM_2D_PVD(),
            examples.load_meshseries_mechanics_3D_XDMF(),
        ],
    )
    def test_transform_order_invariance(self, ms: ot.MeshSeries):
        "Tests that some variable operations are invariant to their order."
        variables = [
            ov.strain.trace.mean.rate(),
            ov.strain.trace.rate().mean,
            ov.strain.mean.trace.rate(),
            ov.strain.mean.rate().trace,
            ov.strain.rate().trace.mean,
            ov.strain.rate().mean.trace,
        ]
        ms = examples.load_meshseries_THM_2D_PVD()
        ref_vals = variables[0].transform(ms)
        for var in variables[1:]:
            # not testing for full equality as the order of operations will
            # cause floating point differences.
            np.testing.assert_allclose(var.transform(ms), ref_vals, rtol=1e-12)

    @pytest.mark.parametrize(
        ("var", "output_type"),
        [
            (ov.temperature.gradient, ov.Vector),
            (ov.stress.von_Mises.gradient, ov.Vector),
            (ov.velocity.divergence, ov.Scalar),
            (ov.velocity.vorticity, ov.Vector),
            (ov.velocity.qcriterion, ov.Scalar),
        ],
    )
    def test_derivatives(self, var: ot.variables.Variable, output_type: type):
        assert isinstance(var, output_type)

    def test_divergence(self):
        "Test approximate equality of strain trace and displacement divergence"
        mesh = examples.load_meshseries_THM_2D_PVD()[-1]
        eps_trace = ov.strain.trace.replace(output_unit="")
        u_div = ov.displacement.divergence
        np.testing.assert_allclose(
            u_div.transform(mesh), eps_trace.transform(mesh), atol=2.3e-5
        )

    def test_divergence_matrix(self):
        "Test divergence of a Matrix correctly reduces to a Vector."
        ms = examples.load_meshseries_THM_2D_PVD()
        var = ov.stress.divergence
        assert var.transform(ms).shape == (len(ms), ms[0].n_points, 3)
        assert var.transform(ms[-1]).shape == (ms[0].n_points, 3)

    def test_derivative_units(self):
        mesh = examples.load_meshseries_THM_2D_PVD()[-1]
        stress_pa = ov.stress.replace(output_unit="Pa")
        div_pa = stress_pa.divergence.transform(mesh)
        div_mpa = ov.stress.divergence.transform(mesh)
        np.testing.assert_allclose(div_mpa, div_pa * 1e-6, 0, 0)

        grad_pa = stress_pa.trace.gradient.transform(mesh)
        grad_mpa = ov.stress.trace.gradient.transform(mesh)
        np.testing.assert_allclose(grad_mpa, grad_pa * 1e-6, 0, 0)

        velocity_mm_s = ov.velocity.replace(output_unit="mm/s")
        div_mm_s = velocity_mm_s.divergence.transform(mesh)
        div_m_s = ov.velocity.divergence.transform(mesh)
        np.testing.assert_allclose(div_m_s, div_mm_s * 1e-3, 1e-15, 0)

    def test_derivative_spatial_unit(self):
        "Test correct handling of spatial unit of dataset."
        ms = examples.load_meshseries_THM_2D_PVD()
        var = ov.temperature.gradient
        ref = var.transform(ms[-1])
        actual = var.transform(ms.scale("km")[-1])
        np.testing.assert_allclose(actual, 1e3 * ref)

    def test_polar_tranformation_2D(self):
        "Check with different formulation for rr and tt components."
        mesh = examples.load_mesh_mechanics_2D()
        sig = mesh["sigma"]
        center = (150, -650, 0)
        phi = angles(mesh, center)
        ref_phi = np.atan2(
            mesh.points[:, 1] - center[1], mesh.points[:, 0] - center[0]
        )
        np.testing.assert_allclose(phi, ref_phi)
        ref_sigma_rr, ref_sigma_tt = (
            (
                0.5 * (sig[:, 0] + sig[:, 1])
                + signum * 0.5 * (sig[:, 0] - sig[:, 1]) * np.cos(2 * phi)
                + signum * sig[:, 3] * np.sin(2 * phi)
            )
            for signum in [1, -1]
        )
        sigma_rr = ov.stress.to_polar(center)["rr"].transform(mesh)
        sigma_tt = ov.stress.to_polar(center)["tt"].transform(mesh)
        np.testing.assert_allclose(sigma_rr, 1e-6 * ref_sigma_rr)
        np.testing.assert_allclose(sigma_tt, 1e-6 * ref_sigma_tt)

    def test_polar_tranformation_3D(self):
        """Check, for homogeneous polar stresses in spherical mesh.

        As the mesh is structured, we can group all transformed values together
        per radius and check if they are all roughly the same.
        """
        mesh = examples.load_mesh_mechanics_3D_sphere()
        radii = np.round(np.linalg.norm(mesh.points, axis=-1), 5)
        unique_r = np.unique(radii)
        vals = ov.stress.to_polar().transform(mesh)
        for r in unique_r:
            x = vals[radii == r]
            assert np.allclose(x, x[0], atol=0.4)

    def test_aggregation(self):
        """Check shape and basic expectations for one example."""
        ms = examples.load_meshseries_CT_2D_XDMF()
        ms_bot = ms.transform(lambda mesh: mesh.clip("z", origin=(0, 0, 0)))
        si = ov.saturation
        np.testing.assert_allclose(si.max.transform(ms), 100, rtol=0.005)
        np.testing.assert_allclose(si.min.transform(ms_bot), 0)
        for var in [si.max, si.min, si.mean, si.median, si.sum, si.var, si.std]:
            assert var.transform(ms_bot).shape == (len(ms),)
        # as the total saturation in the domain increases per timestep
        # (at least for the simulated timeframe) we expect the following
        # quantities to be increasing per step
        for var in [si.mean, si.median, si.sum, si.var, si.std]:
            assert np.all(np.diff(var.transform(ms_bot)) >= 0)

    def test_aggregation_unit_correctness(self):
        "Test handling of temperature unit aggregations"
        var = ov.temperature
        vals = [373.15, 374.15, 375.15]
        assert var.max.transform(vals) == pytest.approx(102.0)
        assert var.mean.transform(vals) == pytest.approx(101.0)
        assert var.min.transform(vals) == pytest.approx(100.0)
        assert var.std.transform(vals) == pytest.approx(0.8164966)
        assert var.var.transform(vals) == pytest.approx(0.6666666)
        assert var.sum.transform(vals) == pytest.approx(303.0)

    def test_matrix_agg_on_single_mesh_shape(self):
        """Aggregation on Matrix/Vector must reduce to per-point, not per-component.

        stress.mean on a single mesh collapses axis=-2 (the points axis) and
        returns shape (n_components,), not (n_points,). This must NOT be used
        for per-point visualization. stress.tensor_mean is the correct choice
        for hydrostatic stress per point.
        """
        mesh = examples.load_mesh_mechanics_2D()
        n_pts = mesh.n_points
        # stress.mean aggregates spatially → shape (n_components,), not (n_pts,)
        assert ov.stress.mean.transform(mesh).shape == (4,)
        # stress.tensor_mean gives hydrostatic stress per point → shape (n_pts,)
        assert ov.stress.tensor_mean.transform(mesh).shape == (n_pts,)

    def test_aggregation_introspection(self, np_func: Callable):
        """Aggregation variables expose their registered reducer and axis."""
        assert ov.saturation.aggregation_variable(np_func).aggregations == 1
        # Vector/matrix variables reduce over the points axis
        assert ov.displacement.max.function.params["axis"] == -2
        assert ov.stress.mean.function.params["axis"] == -2
        # Variables without a registered reducer are not aggregations
        assert ov.saturation.aggregations == 0
        assert ov.saturation.abs.aggregations == 0
        assert ov.saturation.rate("d").aggregations == 0
        assert ov.stress.tensor_mean.aggregations == 0
        # More than one registered reducer is ambiguous
        assert ov.saturation.max.min.aggregations == 2
        # A non-reducer transform preceding the reducer does not disqualify
        assert ov.saturation.rate("d").max.aggregations == 1

    def test_new_aggregations(self):
        """Test functionality of not predefined aggregation functions."""
        ms = examples.load_meshseries_THM_2D_PVD()
        for func in [np.nanmax, np.nanargmax]:
            agg_var = ov.temperature.aggregate(func)
            assert not np.any(np.isnan(agg_var.transform(ms)))

    def test_get_data_mask(self):
        def _check_masked_data(ms_check, actual):
            desired = ms_check.point_data[var.data_name]
            for i, mesh in enumerate(ms_check):
                mask = np.asarray(mesh.ctp(pass_cell_data=False)[var.mask] == 0)
                desired[i, mask] = np.nan
            assert np.array_equal(actual, desired, equal_nan=True)

        ms_a = examples.load_meshseries_THM_2D_PVD()
        var = ov.pressure.replace(output_unit=ov.pressure.data_unit)

        # Mask differs with time
        _check_masked_data(ms_a, var.transform(ms_a))

        # Mask time-invariant
        ms_b = ms_a.copy(deep=True)
        mask_a = ms_a.cell_data[var.mask]
        ms_b.cell_data[var.mask] = np.broadcast_to(mask_a[-1], mask_a.shape)
        _check_masked_data(ms_b, var.transform(ms_b))

    def test_matrix_col_row(self):
        "Test matrix row and column equality due to symmetry."
        var = ov.stress
        for data in [np.arange(4), np.arange(6)]:
            for i in range(3):
                np.testing.assert_array_equal(
                    var.col(i).transform(data), var.row(i).transform(data)
                )

    def test_access_via_str(self):
        "Test accessing variables and their components via a string"
        mesh = examples.load_meshseries_THM_2D_PVD()[-1]
        mesh.point_data["scalar"] = np.zeros(mesh.n_points)
        mesh.point_data["vector"] = np.tile([0, 1], (mesh.n_points, 1))
        mesh.point_data["matrix"] = np.tile([0, 1, 2, 3], (mesh.n_points, 1))

        def equal(name: str, var: ov.Variable) -> None:
            str_var = ov.Variable.find(name, mesh)
            var_ = var.find(var, mesh)
            for key in vars(var_):
                if key == "functions":
                    # a simple comparison like var == str_var would fail due to
                    # the functions being stored in different memory locations
                    continue
                assert getattr(str_var, key) == getattr(var_, key)
            np.testing.assert_equal(
                var_.transform(mesh), str_var.transform(mesh)
            )

        for name, var in [
            ("temperature", ov.temperature),
            ("temperature_gradient", ov.temperature.gradient),
            ("displacement", ov.displacement),
            ("displacement_x", ov.displacement["x"]),
            ("displacement_0", ov.displacement[0]),
            ("sigma", ov.stress),
            ("sigma_yy", ov.stress["yy"]),
            ("sigma_2", ov.stress[2]),
            ("scalar", ov.Scalar("scalar")),
            ("vector", ov.Vector("vector")),
            ("vector_x", ov.Vector("vector")["x"]),
            ("vector_1", ov.Vector("vector")[1]),
            ("matrix", ov.Matrix("matrix")),
            ("matrix_xx", ov.Matrix("matrix")["xx"]),
            ("matrix_3", ov.Matrix("matrix")[3]),
        ]:
            equal(name, var)

    def test_error_msg_via_str(self):
        "Test if meaningful error messages are given upon faulty input."
        mesh = examples.load_meshseries_THM_2D_PVD()[-1]
        failcases = [
            ("", "'' not found in dataset."),
            ("Temperature", "'Temperature' not found in dataset."),
            ("temperature_x", "Scalar 'temperature' has no component x"),
            ("velocity_xx", "Vector index can only be 'x', 'y', 'z' or an int"),
            ("sigma_x", "Matrix index can only be an int or one of"),
        ]  # fmt: skip
        for key, message in failcases:
            with pytest.raises(KeyError, match=message):
                ov.Variable.find(key, mesh)

    def test_incompatible_units(self):
        "Test error handling for incompatible units."
        var = ov.Scalar("temperature", "K", "Pa")
        with pytest.raises(TypeError, match="units are incompatible"):
            var.transform(1)
