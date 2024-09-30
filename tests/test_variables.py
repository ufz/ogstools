"""Unit tests for variables."""

import numpy as np
import pytest
from pint.facets.plain import PlainQuantity

from ogstools import examples
from ogstools import variables as ov

Qty = ov.u_reg.Quantity


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
        assert ov.temperature.get_output_unit() == "°C"
        assert ov.pressure.get_output_unit() == "MPa"
        assert ov.strain.get_output_unit() == "%"

    def test_mask(self):
        """Test mask functionality."""
        assert ov.temperature.get_mask().is_mask()

    def test_get_preset(self):
        """Test find variable function."""
        mesh = examples.load_mesh_mechanics_2D()
        mesh.point_data["scalar"] = mesh["temperature"]
        mesh.point_data["vector"] = mesh["displacement"]
        mesh.point_data["matrix"] = mesh["sigma"]
        assert ov.get_preset("temperature", mesh) == ov.temperature
        assert ov.get_preset("displacement", mesh) == ov.displacement
        assert ov.get_preset("sigma", mesh) == ov.stress
        assert ov.get_preset("scalar", mesh) == ov.Scalar("scalar")
        assert ov.get_preset("vector", mesh) == ov.Vector("vector")
        assert ov.get_preset("matrix", mesh) == ov.Matrix("matrix")
        pytest.raises(KeyError, ov.get_preset, "test", mesh)

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
