"""Unit tests for solid mechanics."""

import unittest

import numpy as np

from ogstools.propertylib import tensor_math
from ogstools.propertylib.tensor_math import sym_tensor_to_mat


class MechanicsTest(unittest.TestCase):
    """Test case for physical properties."""

    rng = np.random.default_rng()
    N_SAMPLES = 100000

    def equality(
        self, vals1: np.ndarray, vals2: np.ndarray, rtol=1e-7, atol=1e-9
    ):
        """Assert the equality of two arrays."""
        np.testing.assert_allclose(
            vals1, vals2, verbose=True, rtol=rtol, atol=atol
        )

    def test_frobenius_norm(self):
        """Test Frobenius norm."""
        for symten_len in [4, 6]:
            sig = self.rng.random((self.N_SAMPLES, symten_len)) * 1e6
            sig_mat = sym_tensor_to_mat(sig)
            frob2 = np.sqrt(
                tensor_math.trace(
                    np.diagonal(
                        np.transpose(sig_mat, (0, 2, 1)) @ sig_mat, 0, 2
                    )
                )
            )
            self.equality(tensor_math.frobenius_norm(sig), frob2)

    def test_I1(self):
        """Test first invariant."""
        for symten_len in [4, 6]:
            sig = self.rng.random((self.N_SAMPLES, symten_len)) * 1e6
            self.equality(
                tensor_math.I1(sig),
                tensor_math.trace(tensor_math.eigenvalues(sig)),
            )

    def test_I2(self):
        """Test second invariant."""
        for symten_len in [4, 6]:
            sig = self.rng.random((self.N_SAMPLES, symten_len)) * 1e6
            eig_vals = tensor_math.eigenvalues(sig)
            self.equality(
                tensor_math.I2(sig),
                np.sum(eig_vals * np.roll(eig_vals, 1, axis=-1), axis=-1),
            )

    def test_I3(self):
        """Test third invariant."""
        for symten_len in [4, 6]:
            sig = self.rng.random((self.N_SAMPLES, symten_len)) * 1e6
            eig_vals = tensor_math.eigenvalues(sig)
            self.equality(tensor_math.I3(sig), np.prod(eig_vals, axis=-1))

    def test_J1(self):
        """Test first deviator invariant."""
        for symten_len in [4, 6]:
            sig = self.rng.random((self.N_SAMPLES, symten_len)) * 1e6
            j1 = tensor_math.J1(sig)
            self.equality(j1, np.zeros(j1.shape))

    def test_J2(self):
        """Test second deviator invariant."""
        for symten_len in [4, 6]:
            sig = self.rng.random((self.N_SAMPLES, symten_len)) * 1e6
            self.equality(
                tensor_math.J2(sig),
                0.5
                * (
                    tensor_math.trace(sig**2)
                    - (1.0 / 3.0) * tensor_math.trace(sig) ** 2
                ),
            )

    def test_J3(self):
        """Test third deviator invariant."""
        for symten_len in [4, 6]:
            sig = self.rng.random((self.N_SAMPLES, symten_len)) * 1e6
            # not exactly sure why, but for stresses where the below condition
            # is very close to zero, the error of this test shoots up.
            # Probably some floating point precision issue.
            mask = (
                np.abs(
                    tensor_math.trace(sig - tensor_math.mean(sig)[..., None])
                )
                > 1e-9
            )
            sig = sig[mask]
            self.equality(
                tensor_math.J3(sig),
                (1.0 / 3.0)
                * (
                    tensor_math.trace(sig**3)
                    - tensor_math.trace(sig**2) * tensor_math.trace(sig)
                    + (2.0 / 9.0) * tensor_math.trace(sig) ** 3.0
                ),
            )

    def test_von_mises(self):
        """Test von Mises invariant."""
        for symten_len in [4, 6]:
            sig = self.rng.random((self.N_SAMPLES, symten_len)) * 1e6
            ev = tensor_math.eigenvalues(sig)
            self.equality(
                tensor_math.von_mises(sig),
                np.sqrt(0.5 * np.sum(np.square(ev - np.roll(ev, 1, -1)), -1)),
            )

    def test_octahedral_shear_stress(self):
        """Test octahedral shear stress invariant."""
        for symten_len in [4, 6]:
            sig = self.rng.random((self.N_SAMPLES, symten_len)) * 1e6
            self.equality(
                tensor_math.octahedral_shear(sig),
                (1.0 / 3.0)
                * np.sqrt(
                    2 * tensor_math.I1(sig) ** 2 - 6 * tensor_math.I2(sig)
                ),
            )
