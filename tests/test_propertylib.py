"""Unit tests for physical properties."""

import unittest

import numpy as np
from pint.facets.plain import PlainQuantity

from ogstools.propertylib import presets as _p
from ogstools.propertylib.property import Scalar, u_reg

Q_ = u_reg.Quantity
EPS = 1e-7


class PhysicalPropertyTest(unittest.TestCase):
    """Test case for physical properties."""

    def equality(self, p: Scalar, vals: np.ndarray, res: PlainQuantity):
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
        self.equality(_p.temperature, 293.15, Q_(20, "°C"))
        self.equality(_p.temperature, [293.15, 303.15], Q_([20, 30], "°C"))

    def test_pressure(self):
        """Test pressure property."""
        self.equality(_p.pressure, 1e6, Q_(1, "MPa"))
        self.equality(_p.pressure, [1e6, 2e6], Q_([1, 2], "MPa"))

    def test_velocity(self):
        """Test velocity property."""
        self.equality(_p.velocity.magnitude, [3, 4], Q_(5, "m/s"))
        self.equality(
            _p.velocity.magnitude, [[3, 4], [1, 0]], Q_([5, 1], "m/s")
        )
        self.equality(_p.velocity.log_magnitude, np.sqrt([50, 50]), Q_(1, ""))
        self.equality(
            _p.velocity.log_magnitude,
            [np.sqrt([50, 50]), [10, 0]],
            Q_([1, 1], ""),
        )

    def test_displacement(self):
        """Test displacement property."""
        self.equality(_p.displacement.magnitude, [3e-3, 4e-3], Q_(5, "mm"))
        u = np.array([[2, 3, 6], [1, 4, 8]]) * 1e-3
        self.equality(_p.displacement.magnitude, u, Q_([7.0, 9.0], "mm"))

    def test_strain(self):
        """Test strain property."""
        eps = np.array([1, 3, 9, 1, 2, 2]) * 1e-2
        self.equality(_p.strain.magnitude, eps, Q_(10, "%"))
        self.equality(_p.strain.magnitude, [eps, eps], Q_([10, 10], "%"))
        self.equality(_p.strain.trace, eps, Q_(13, "%"))
        self.equality(_p.strain.trace, [eps, eps], Q_([13, 13], "%"))

    def test_components(self):
        """Test strain components."""
        eps = np.array([0, 1, 2, 3, 4, 5]) * 1e-2
        u = np.array([0, 1, 2]) * 1e-3
        for i in range(len(eps)):
            self.equality(_p.strain[i], eps, Q_(i, "%"))
            self.equality(_p.strain[i], [eps, eps], Q_([i, i], "%"))
        for i in range(len(u)):
            self.equality(_p.displacement[i], u, Q_(i, "mm"))
            self.equality(_p.displacement[i], [u, u], Q_([i, i], "mm"))
        assert _p.strain[0].bilinear_cmap is True

    def test_von_mises(self):
        """Test von_mises_stress property."""
        sig_3D = np.array([4, 1, 2, 1, 1, 1]) * 1e6
        self.equality(_p.von_mises_stress, sig_3D, Q_(4, "MPa"))
        self.equality(_p.von_mises_stress, [sig_3D, sig_3D], Q_([4, 4], "MPa"))
        sig_2D = np.array([4, 1, 2, 3**0.5]) * 1e6
        self.equality(_p.von_mises_stress, sig_2D, Q_(4, "MPa"))

    def test_eff_pressure(self):
        """Test effective_pressure property."""
        sig = np.array([-1, -2, -3, 1, 1, 1]) * 1e6
        self.equality(_p.effective_pressure, sig, Q_(2, "MPa"))
        self.equality(_p.effective_pressure, [sig, sig], Q_([2, 2], "MPa"))

    def test_qp_ratio(self):
        """Test qp_ratio property."""
        sig = np.array([4, 1, 2, 1, 1, 1]) * 1e6
        self.equality(_p.qp_ratio, sig, Q_(-100 * 12 / 7, "percent"))
        self.equality(
            _p.qp_ratio, [sig] * 2, Q_([-100 * 12 / 7] * 2, "percent")
        )

    def test_simple(self):
        """Test call functionality."""
        self.assertEqual(_p.temperature(273.15), Q_(0, "°C"))
        self.assertEqual(_p.displacement[0]([1, 2, 3]), Q_(1, "m"))
        self.assertEqual(_p.displacement([1, 2, 3])[1], Q_(2, "m"))

    def test_values(self):
        """Test values functionality."""
        self.assertEqual(_p.temperature.strip_units(273.15), 0.0)

    def test_units(self):
        """Test get_output_unit functionality."""
        self.assertEqual(_p.temperature.get_output_unit(), "°C")
        self.assertEqual(_p.pressure.get_output_unit(), "MPa")
        self.assertEqual(_p.strain.get_output_unit(), "%")

    def test_mask(self):
        """Test mask functionality."""
        self.assertTrue(_p.temperature.get_mask().is_mask())

    def test_find_property(self):
        """Test find property function."""
        self.assertEqual(_p.find_property("pressure"), _p.pressure)

    def test_copy_ctor(self):
        """Test replace constructor."""

        strain_copy = _p.stress.replace(
            data_name=_p.strain.data_name,
            data_unit=_p.strain.data_unit,
            output_unit=_p.strain.output_unit,
            output_name=_p.strain.output_name,
            mask=_p.strain.mask,
            func=_p.strain.func,
        )

        self.assertEqual(_p.strain, strain_copy)


if __name__ == "__main__":
    unittest.main(argv=[""], verbosity=2, exit=False)
