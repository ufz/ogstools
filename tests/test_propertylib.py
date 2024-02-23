"""Unit tests for physical properties."""

import unittest

import numpy as np
from pint.facets.plain import PlainQuantity

from ogstools.propertylib import presets as pp
from ogstools.propertylib.property import Scalar, u_reg

qty = u_reg.Quantity


class PhysicalPropertyTest(unittest.TestCase):
    """Test case for physical properties."""

    EPS = 1e-7

    def equality(self, p: Scalar, vals: np.ndarray, res: PlainQuantity):
        """
        Assert the equality of property calculations.

        :param property: The scalar property to test.
        :param values: The input values.
        :param expected_result: The expected result.
        """
        np.testing.assert_allclose(
            p(vals),
            res.to(p.output_unit).magnitude,
            rtol=self.EPS,
            verbose=True,
        )

    def test_temperature(self):
        """Test temperature property."""
        self.equality(pp.temperature, 293.15, qty(20, "°C"))
        self.equality(pp.temperature, [293.15, 303.15], qty([20, 30], "°C"))

    def test_pressure(self):
        """Test pressure property."""
        self.equality(pp.pressure, 1e6, qty(1, "MPa"))
        self.equality(pp.pressure, [1e6, 2e6], qty([1, 2], "MPa"))

    def test_velocity(self):
        """Test velocity property."""
        self.equality(pp.velocity.magnitude, [3, 4], qty(5, "m/s"))
        self.equality(
            pp.velocity.magnitude, [[3, 4], [1, 0]], qty([5, 1], "m/s")
        )

    def test_displacement(self):
        """Test displacement property."""
        self.equality(pp.displacement.magnitude, [3e-3, 4e-3], qty(5, "mm"))
        u = np.array([[2, 3, 6], [1, 4, 8]]) * 1e-3
        self.equality(pp.displacement.magnitude, u, qty([7.0, 9.0], "mm"))

    def test_strain(self):
        """Test strain property."""
        eps = np.array([1, 3, 9, 1, 2, 2]) * 1e-2
        self.equality(pp.strain.magnitude, eps, qty(10, "%"))
        self.equality(pp.strain.magnitude, [eps, eps], qty([10, 10], "%"))
        self.equality(pp.strain.trace, eps, qty(13, "%"))
        self.equality(pp.strain.trace, [eps, eps], qty([13, 13], "%"))

    def test_components(self):
        """Test strain components."""
        eps = np.array([0, 1, 2, 3, 4, 5]) * 1e-2
        u = np.array([0, 1, 2]) * 1e-3
        for i in range(len(eps)):
            self.equality(pp.strain[i], eps, qty(i, "%"))
            self.equality(pp.strain[i], [eps, eps], qty([i, i], "%"))
        for i in range(len(u)):
            self.equality(pp.displacement[i], u, qty(i, "mm"))
            self.equality(pp.displacement[i], [u, u], qty([i, i], "mm"))
        assert pp.strain[0].bilinear_cmap is True

    def test_von_mises(self):
        """Test von_mises_stress property."""
        sig_3D = np.array([3, 1, 1, 1, 1, 1]) * 1e6
        self.equality(pp.von_mises_stress, sig_3D, qty(2, "MPa"))
        self.equality(pp.von_mises_stress, [sig_3D, sig_3D], qty([2, 2], "MPa"))
        sig_2D = np.array([2, 1, 1, 1]) * 1e6
        self.equality(pp.von_mises_stress, sig_2D, qty(1, "MPa"))

    def test_eff_pressure(self):
        """Test effective_pressure property."""
        sig = np.array([-1, -2, -3, 1, 1, 1]) * 1e6
        self.equality(pp.effective_pressure, sig, qty(2, "MPa"))
        self.equality(pp.effective_pressure, [sig, sig], qty([2, 2], "MPa"))

    def test_qp_ratio(self):
        """Test qp_ratio property."""
        sig = np.array([4, 4, 1, 1, 1, 1]) * 1e6
        self.equality(pp.qp_ratio, sig, qty(-100, "percent"))
        self.equality(pp.qp_ratio, [sig] * 2, qty([-100] * 2, "percent"))

    def test_dilatancy(self):
        """Test dilatancy property."""
        sig = np.array([4, 1, 2, 1, 1, 1]) * 1e6
        self.assertRaises(TypeError, pp.dilatancy_alkan_eff, sig)

    def test_simple(self):
        """Test call functionality."""
        self.assertEqual(pp.temperature(273.15, strip_unit=False), qty(0, "°C"))
        self.assertEqual(
            pp.displacement[0]([1, 2, 3], strip_unit=False), qty(1, "m")
        )
        self.assertEqual(
            pp.displacement([1, 2, 3], strip_unit=False)[1], qty(2, "m")
        )

    def test_values(self):
        """Test values functionality."""
        self.assertEqual(pp.temperature(273.15), 0.0)

    def test_units(self):
        """Test get_output_unit functionality."""
        self.assertEqual(pp.temperature.get_output_unit(), "°C")
        self.assertEqual(pp.pressure.get_output_unit(), "MPa")
        self.assertEqual(pp.strain.get_output_unit(), "%")

    def test_mask(self):
        """Test mask functionality."""
        self.assertTrue(pp.temperature.get_mask().is_mask())

    def test_get_preset(self):
        """Test find property function."""
        self.assertEqual(pp.get_preset("pressure"), pp.pressure)
        self.assertEqual(pp.get_preset("pore_pressure"), pp.pressure)
        self.assertEqual(pp.get_preset("test"), Scalar("test"))
        self.assertEqual(pp.get_preset("test", shape=(100, 3)), Vector("test"))
        self.assertEqual(pp.get_preset("test", shape=(100, 6)), Matrix("test"))

    def test_copy_ctor(self):
        """Test replace constructor."""

        strain_copy = pp.stress.replace(
            data_name=pp.strain.data_name,
            data_unit=pp.strain.data_unit,
            output_unit=pp.strain.output_unit,
            output_name=pp.strain.output_name,
            mask=pp.strain.mask,
            func=pp.strain.func,
        )

        self.assertEqual(pp.strain, strain_copy)
