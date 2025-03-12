"""Unit tests for variables."""

from itertools import pairwise

import numpy as np
import pytest
from pint.facets.plain import PlainQuantity

from ogstools import examples
from ogstools import variables as ov

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
        sig = np.array([4, 1, 2, 1, 1, 1]) * 1e6
        #  not working for arrays (only works for meshes)
        pytest.raises(TypeError, ov.dilatancy_alkan.transform, sig)
        mesh = examples.load_mesh_mechanics_2D()
        mesh.point_data["pressure"] = mesh.p_fluid()
        assert np.max(ov.dilatancy_alkan.transform(mesh)) > 0
        assert np.max(ov.dilatancy_alkan_eff.transform(mesh)) > 0
        assert np.max(ov.dilatancy_critescu_tot.transform(mesh)) > 0
        assert np.max(ov.dilatancy_critescu_eff.transform(mesh)) > 0
        assert np.max(ov.fluid_pressure_crit.transform(mesh)) > 0

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
        assert ov.stress.mean.transform(sig) > 0
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
        pytest.raises(KeyError, ov.Variable.find, "test", mesh)
        # testcase with str matching output_name of the predefined Variables
        mesh = examples.load_meshseries_HT_2D_XDMF()[0]
        darcy = ov.Variable.find("darcy_velocity", mesh)
        assert darcy.data_name in mesh.point_data

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
        for _, components in ov.BHE_Vector.BHE_COMPONENTS.items():
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

    @pytest.mark.parametrize(
        "bhe_type", list(ov.temperature_BHE.BHE_COMPONENTS)
    )
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
        index_combinations = index_gen(ov.BHE_Vector.BHE_COMPONENTS[bhe_type])
        temp = [
            ms.probe((0, 0, 0), ov.temperature_BHE[1, idx])[0]
            for idx in index_combinations
        ]
        # initial vector is +1 for every component
        assert np.all(np.diff(temp) == 1)

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
            assert vmin <= np.min(vals)
            assert vmax >= np.max(vals)

        check_limits(ov.temperature, 7.9, 50)
        check_limits(ov.pressure, 0, 8)
        check_limits(ov.displacement.magnitude, 0, 0.3)
