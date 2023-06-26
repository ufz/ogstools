"""Unit tests for physical properties."""

import unittest

import numpy as np

from ogstools.propertylib import HM, TH, THM, TM, H, M, T, property_collection
from ogstools.propertylib.property import ScalarProperty, u_reg

Q_ = u_reg.Quantity
EPS = 1e-7


class PhysicalPropertyTest(unittest.TestCase):
    """Test case for physical properties."""

    def equality(self, p: ScalarProperty, vals: np.ndarray, res):
        """
        Assert the equality of property calculations.

        :param property: The scalar property to test.
        :param values: The input values.
        :param expected_result: The expected result.
        """
        print(p.quantity(vals))
        print(res)
        np.testing.assert_allclose(
            p.quantity(vals).magnitude,
            res.to(p.output_unit).magnitude,
            rtol=EPS,
            verbose=True,
        )

    def test_temperature(self):
        """Test temperature property."""
        self.equality(T.temperature, 293.15, Q_(20, "°C"))

    def test_pressure(self):
        """Test pressure property."""
        self.equality(H.pressure, 1e6, Q_(1, "MPa"))

    def test_velocity(self):
        """Test qp_ratio property."""
        self.equality(H.velocity.log_magnitude, np.sqrt([50, 50]), Q_(1, ""))
        self.equality(H.velocity.trace, np.array([1.0, 2.0]), Q_(3.0, "m/s"))

    def test_displacement(self):
        """Test displacement property."""
        u = np.array([[2, 3, 6], [1, 4, 8]]) * 1e-3
        self.equality(M.displacement.magnitude, u, Q_([7.0, 9.0], "mm"))

    def test_strain(self):
        """Test strain property."""
        eps = np.array([1, 3, 9, 1, 2, 2]) * 1e-2
        self.equality(M.strain.trace, eps, Q_(13, "percent"))

    def test_components(self):
        """Test strain components."""
        eps = np.array([0, 1, 2, 3, 4, 5]) * 1e-2
        u = np.array([0, 1, 2]) * 1e-3
        for i in range(len(eps)):
            self.equality(M.strain[i], eps, Q_(i, "percent"))
            self.equality(M.strain.component(i), eps, Q_(i, "percent"))
        for i in range(len(u)):
            self.equality(M.displacement[i], u, Q_(i, "mm"))
            self.equality(M.displacement.component(i), u, Q_(i, "mm"))
        assert M.strain.component(0).is_component

    def test_von_mises(self):
        """Test von_mises_stress property."""
        self.equality(
            M.von_mises_stress, np.array([4, 1, 2, 1, 1, 1]) * 1e6, Q_(4, "MPa")
        )
        self.equality(
            M.von_mises_stress,
            np.array([4, 1, 2, 3**0.5]) * 1e6,
            Q_(4, "MPa"),
        )

    def test_eff_pressure(self):
        """Test effective_pressure property."""
        self.equality(
            M.effective_pressure,
            np.array([-1, -2, -3, 1, 1, 1]) * 1e6,
            Q_(2, "MPa"),
        )

    def test_qp_ratio(self):
        """Test qp_ratio property."""
        self.equality(
            M.qp_ratio,
            np.array([4, 1, 2, 1, 1, 1]) * 1e6,
            Q_(-100 * 12 / 7, "percent"),
        )

    def test_cast(self):
        """Test cast functionality."""
        assert T.temperature.cast(273.15) == Q_(0, "°C")
        assert M.displacement[0].cast([1, 2, 3]) == Q_(1, "m")
        assert M.displacement.cast([1, 2, 3])[1] == Q_(2, "m")

    def test_values(self):
        """Test values functionality."""
        assert T.temperature.values(273.15) == 0.0

    def test_units(self):
        """Test get_output_unit functionality."""
        assert T.temperature.get_output_unit() == "°C"
        assert H.pressure.get_output_unit() == "MPa"
        assert M.strain.get_output_unit() == "%"

    def test_mask(self):
        """Test get_output_unit functionality."""
        assert ScalarProperty("pressure_active").is_component

    def test_processes(self):
        """Test process attributes."""

        def data_name_set(process: property_collection.PropertyCollection):
            return {p.data_name for p in process.get_properties()}

        assert data_name_set(TH) == data_name_set(T) | data_name_set(H)
        assert data_name_set(HM) == data_name_set(H) | data_name_set(M)
        assert data_name_set(TM) == data_name_set(T) | data_name_set(M)
        assert data_name_set(THM) == data_name_set(TH) | data_name_set(M)

    def test_copy_ctor(self):
        """Test process attributes."""

        strain_copy = M.stress(
            data_name=M.strain.data_name,
            data_unit=M.strain.data_unit,
            output_unit=M.strain.output_unit,
            output_name=M.strain.output_name,
            mask=M.strain.mask,
            func=M.strain.func,
        )

        assert M.strain == strain_copy


if __name__ == "__main__":
    unittest.main(argv=[""], verbosity=2, exit=False)
