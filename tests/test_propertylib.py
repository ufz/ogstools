"""Unit tests for physical properties."""

import unittest

import numpy as np
import pyvista as pv
from pint.facets.plain import PlainQuantity

from ogstools import examples
from ogstools.propertylib import properties as pp
from ogstools.propertylib.mesh_dependent import depth
from ogstools.propertylib.property import u_reg

Qty = u_reg.Quantity


class PhysicalPropertyTest(unittest.TestCase):
    """Test case for physical properties."""

    EPS = 1e-7

    def equality(self, p: pp.Scalar, vals: np.ndarray, res: PlainQuantity):
        """
        Assert the equality of property calculations.

        :param property: The scalar property to test.
        :param values: The input values.
        :param expected_result: The expected result.
        """
        np.testing.assert_allclose(
            p.transform(vals),
            res.to(p.output_unit).magnitude,
            rtol=self.EPS,
            verbose=True,
        )

    def test_temperature(self):
        """Test temperature property."""
        self.equality(pp.temperature, 293.15, Qty(20, "°C"))
        self.equality(pp.temperature, [293.15, 303.15], Qty([20, 30], "°C"))

    def test_pressure(self):
        """Test pressure property."""
        self.equality(pp.pressure, 1e6, Qty(1, "MPa"))
        self.equality(pp.pressure, [1e6, 2e6], Qty([1, 2], "MPa"))

    def test_velocity(self):
        """Test velocity property."""
        self.equality(pp.velocity.magnitude, [3, 4], Qty(5, "m/s"))
        self.equality(
            pp.velocity.magnitude, [[3, 4], [1, 0]], Qty([5, 1], "m/s")
        )

    def test_displacement(self):
        """Test displacement property."""
        self.equality(pp.displacement.magnitude, [3e-3, 4e-3], Qty(5, "mm"))
        u = np.array([[2, 3, 6], [1, 4, 8]]) * 1e-3
        self.equality(pp.displacement.magnitude, u, Qty([7.0, 9.0], "mm"))

    def test_strain(self):
        """Test strain property."""
        eps = np.array([1, 3, 9, 1, 2, 2]) * 1e-2
        self.equality(pp.strain.magnitude, eps, Qty(10, "%"))
        self.equality(pp.strain.magnitude, [eps, eps], Qty([10, 10], "%"))
        self.equality(pp.strain.trace, eps, Qty(13, "%"))
        self.equality(pp.strain.trace, [eps, eps], Qty([13, 13], "%"))

    def test_components(self):
        """Test strain components."""
        eps = np.array([0, 1, 2, 3, 4, 5]) * 1e-2
        u = np.array([0, 1, 2]) * 1e-3
        for i in range(len(eps)):
            self.equality(pp.strain[i], eps, Qty(i, "%"))
            self.equality(pp.strain[i], [eps, eps], Qty([i, i], "%"))
        for i in range(len(u)):
            self.equality(pp.displacement[i], u, Qty(i, "mm"))
            self.equality(pp.displacement[i], [u, u], Qty([i, i], "mm"))
        assert pp.strain[0].bilinear_cmap is True

    def test_von_mises(self):
        """Test von_mises_stress property."""
        sig_3D = np.array([3, 1, 1, 1, 1, 1]) * 1e6
        self.equality(pp.stress.von_Mises, sig_3D, Qty(2, "MPa"))
        self.equality(pp.stress.von_Mises, [sig_3D, sig_3D], Qty([2, 2], "MPa"))
        sig_2D = np.array([2, 1, 1, 1]) * 1e6
        self.equality(pp.stress.von_Mises, sig_2D, Qty(1, "MPa"))

    def test_eff_pressure(self):
        """Test effective_pressure property."""
        sig = np.array([-1, -2, -3, 1, 1, 1]) * 1e6
        self.equality(pp.effective_pressure, sig, Qty(2, "MPa"))
        self.equality(pp.effective_pressure, [sig, sig], Qty([2, 2], "MPa"))

    def test_qp_ratio(self):
        """Test qp_ratio property."""
        sig = np.array([4, 4, 1, 1, 1, 1]) * 1e6
        self.equality(pp.stress.qp_ratio, sig, Qty(-100, "percent"))
        self.equality(pp.stress.qp_ratio, [sig] * 2, Qty([-100] * 2, "percent"))

    def test_depth_2D(self):
        mesh = examples.load_mesh_mechanics_2D()
        mesh["depth"] = depth(mesh, use_coords=True)
        # y Axis is vertical axis
        self.assertTrue(np.all(mesh["depth"] == -mesh.points[..., 1]))
        mesh["depth"] = depth(mesh)
        self.assertTrue(np.all(mesh["depth"] < -mesh.points[..., 1]))

    def test_depth_3D(self):
        mesh = pv.SolidSphere(100, center=(0, 0, -101))
        mesh["depth"] = depth(mesh, use_coords=True)
        self.assertTrue(np.all(mesh["depth"] == -mesh.points[..., -1]))
        mesh["depth"] = depth(mesh)
        self.assertTrue(np.all(mesh["depth"] < -mesh.points[..., -1]))

    def test_integrity_criteria(self):
        """Test integrity criteria."""
        sig = np.array([4, 1, 2, 1, 1, 1]) * 1e6
        #  not working for arrays (only works for meshes)
        self.assertRaises(TypeError, pp.dilatancy_alkan.transform, sig)
        mesh = examples.load_mesh_mechanics_2D()
        self.assertGreater(np.max(pp.dilatancy_alkan.transform(mesh)), 0)
        self.assertGreater(np.max(pp.dilatancy_alkan_eff.transform(mesh)), 0)
        self.assertGreater(np.max(pp.dilatancy_critescu_tot.transform(mesh)), 0)
        self.assertGreater(np.max(pp.dilatancy_critescu_eff.transform(mesh)), 0)
        self.assertGreater(np.max(pp.fluid_pressure_crit.transform(mesh)), 0)

    def test_tensor_attributes(self):
        """Test that the access of tensor attributes works."""
        # data needs to be a 2D array
        sig = np.asarray([[4, 1, 2, 1, 1, 1]]) * 1e6
        self.assertTrue(np.all(pp.stress.eigenvalues.transform(sig) >= 0))
        for i in range(3):
            np.testing.assert_allclose(
                pp.stress.eigenvectors[i].magnitude.transform(sig), 1.0
            )
        self.assertGreater(pp.stress.det.transform(sig), 0)
        self.assertGreater(pp.stress.invariant_1.transform(sig), 0)
        self.assertGreater(pp.stress.invariant_2.transform(sig), 0)
        self.assertGreater(pp.stress.invariant_3.transform(sig), 0)
        self.assertGreater(pp.stress.mean.transform(sig), 0)
        self.assertGreater(pp.stress.deviator.magnitude.transform(sig), 0)
        self.assertGreater(pp.stress.deviator_invariant_1.transform(sig), 0)
        self.assertGreater(pp.stress.deviator_invariant_2.transform(sig), 0)
        self.assertGreater(pp.stress.deviator_invariant_3.transform(sig), 0)
        self.assertGreater(pp.stress.octahedral_shear.transform(sig), 0)

    def test_simple(self):
        """Test call functionality."""
        self.assertEqual(
            pp.temperature.transform(273.15, strip_unit=False), Qty(0, "°C")
        )
        self.assertEqual(
            pp.displacement[0].transform([1, 2, 3], strip_unit=False),
            Qty(1, "m"),
        )
        self.assertEqual(
            pp.displacement.transform([1, 2, 3], strip_unit=False)[1],
            Qty(2, "m"),
        )

    def test_values(self):
        """Test values functionality."""
        self.assertEqual(pp.temperature.transform(273.15), 0.0)

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
        mesh = examples.load_mesh_mechanics_2D()
        mesh.point_data["scalar"] = mesh["temperature"]
        mesh.point_data["vector"] = mesh["displacement"]
        mesh.point_data["matrix"] = mesh["sigma"]
        self.assertEqual(pp.get_preset("temperature", mesh), pp.temperature)
        self.assertEqual(pp.get_preset("displacement", mesh), pp.displacement)
        self.assertEqual(pp.get_preset("sigma", mesh), pp.stress)
        self.assertEqual(pp.get_preset("scalar", mesh), pp.Scalar("scalar"))
        self.assertEqual(pp.get_preset("vector", mesh), pp.Vector("vector"))
        self.assertEqual(pp.get_preset("matrix", mesh), pp.Matrix("matrix"))
        self.assertRaises(KeyError, pp.get_preset, "test", mesh)

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

    def test_get_label(self):
        self.assertEqual(pp.pressure.get_label(), "pore pressure / MPa")
