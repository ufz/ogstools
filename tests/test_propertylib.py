"""Unit tests for physical properties."""

import unittest

import numpy as np
from pint.facets.plain import PlainQuantity

from ogstools.propertylib import HM, TH, THM, TM, H, M, PropertyCollection, T
from ogstools.propertylib.property import ScalarProperty, u_reg

Q_ = u_reg.Quantity
EPS = 1e-7


class PhysicalPropertyTest(unittest.TestCase):
    """Test case for physical properties."""

    def equality(self, p: ScalarProperty, vals: np.ndarray, res: PlainQuantity):
        """
        Assert the equality of property calculations.

        :param property: The scalar property to test.
        :param values: The input values.
        :param expected_result: The expected result.
        """
        np.testing.assert_allclose(
            p(vals).magnitude,
            res.to(p.output_unit).magnitude,
            rtol=EPS,
            verbose=True,
        )

    def test_temperature(self):
        """Test temperature property."""
        self.equality(T.temperature, 293.15, Q_(20, "°C"))
        self.equality(T.temperature, [293.15, 303.15], Q_([20, 30], "°C"))

    def test_pressure(self):
        """Test pressure property."""
        self.equality(H.pressure, 1e6, Q_(1, "MPa"))
        self.equality(H.pressure, [1e6, 2e6], Q_([1, 2], "MPa"))

    def test_velocity(self):
        """Test velocity property."""
        self.equality(H.velocity.magnitude, [3, 4], Q_(5, "m/s"))
        self.equality(H.velocity.magnitude, [[3, 4], [1, 0]], Q_([5, 1], "m/s"))
        self.equality(H.velocity.log_magnitude, np.sqrt([50, 50]), Q_(1, ""))
        self.equality(
            H.velocity.log_magnitude,
            [np.sqrt([50, 50]), [10, 0]],
            Q_([1, 1], ""),
        )

    def test_displacement(self):
        """Test displacement property."""
        self.equality(M.displacement.magnitude, [3e-3, 4e-3], Q_(5, "mm"))
        u = np.array([[2, 3, 6], [1, 4, 8]]) * 1e-3
        self.equality(M.displacement.magnitude, u, Q_([7.0, 9.0], "mm"))

    def test_strain(self):
        """Test strain property."""
        eps = np.array([1, 3, 9, 1, 2, 2]) * 1e-2
        self.equality(M.strain.magnitude, eps, Q_(10, "%"))
        self.equality(M.strain.magnitude, [eps, eps], Q_([10, 10], "%"))
        self.equality(M.strain.trace, eps, Q_(13, "%"))
        self.equality(M.strain.trace, [eps, eps], Q_([13, 13], "%"))

    def test_components(self):
        """Test strain components."""
        eps = np.array([0, 1, 2, 3, 4, 5]) * 1e-2
        u = np.array([0, 1, 2]) * 1e-3
        for i in range(len(eps)):
            self.equality(M.strain[i], eps, Q_(i, "%"))
            self.equality(M.strain[i], [eps, eps], Q_([i, i], "%"))
        for i in range(len(u)):
            self.equality(M.displacement[i], u, Q_(i, "mm"))
            self.equality(M.displacement[i], [u, u], Q_([i, i], "mm"))
        assert M.strain[0].is_component

    def test_von_mises(self):
        """Test von_mises_stress property."""
        sig_3D = np.array([4, 1, 2, 1, 1, 1]) * 1e6
        self.equality(M.von_mises_stress, sig_3D, Q_(4, "MPa"))
        self.equality(M.von_mises_stress, [sig_3D, sig_3D], Q_([4, 4], "MPa"))
        sig_2D = np.array([4, 1, 2, 3**0.5]) * 1e6
        self.equality(M.von_mises_stress, sig_2D, Q_(4, "MPa"))

    def test_eff_pressure(self):
        """Test effective_pressure property."""
        sig = np.array([-1, -2, -3, 1, 1, 1]) * 1e6
        self.equality(M.effective_pressure, sig, Q_(2, "MPa"))
        self.equality(M.effective_pressure, [sig, sig], Q_([2, 2], "MPa"))

    def test_qp_ratio(self):
        """Test qp_ratio property."""
        sig = np.array([4, 1, 2, 1, 1, 1]) * 1e6
        self.equality(M.qp_ratio, sig, Q_(-100 * 12 / 7, "percent"))
        self.equality(M.qp_ratio, [sig] * 2, Q_([-100 * 12 / 7] * 2, "percent"))

    def test_simple(self):
        """Test call functionality."""
        self.assertEqual(T.temperature(273.15), Q_(0, "°C"))
        self.assertEqual(M.displacement[0]([1, 2, 3]), Q_(1, "m"))
        self.assertEqual(M.displacement([1, 2, 3])[1], Q_(2, "m"))

    def test_values(self):
        """Test values functionality."""
        self.assertEqual(T.temperature.values(273.15), 0.0)

    def test_units(self):
        """Test get_output_unit functionality."""
        self.assertEqual(T.temperature.get_output_unit(), "°C")
        self.assertEqual(H.pressure.get_output_unit(), "MPa")
        self.assertEqual(M.strain.get_output_unit(), "%")

    def test_mask(self):
        """Test mask functionality."""
        self.assertTrue(T.temperature.get_mask().is_mask())

    def test_processes(self):
        """Test process attributes."""

        def data_names(process: PropertyCollection):
            return {p.data_name for p in process.get_properties()}

        self.assertEqual(data_names(TH), data_names(T) | data_names(H))
        self.assertEqual(data_names(HM), data_names(H) | data_names(M))
        self.assertEqual(data_names(TM), data_names(T) | data_names(M))
        self.assertEqual(data_names(THM), data_names(TH) | data_names(M))

    def test_copy_ctor(self):
        """Test replace constructor."""

        strain_copy = M.stress.replace(
            data_name=M.strain.data_name,
            data_unit=M.strain.data_unit,
            output_unit=M.strain.output_unit,
            output_name=M.strain.output_name,
            mask=M.strain.mask,
            func=M.strain.func,
        )

        self.assertEqual(M.strain, strain_copy)


if __name__ == "__main__":
    unittest.main(argv=[""], verbosity=2, exit=False)
