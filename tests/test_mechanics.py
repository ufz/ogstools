"""Unit tests for solid mechanics."""

import numpy as np
import pytest

from ogstools.variables import tensor_math


def assert_allclose(vals1: np.ndarray, vals2: np.ndarray, rtol=1e-5, atol=1e-9):
    """Assert the equality of two arrays."""
    np.testing.assert_allclose(vals1, vals2, verbose=True, rtol=rtol, atol=atol)


@pytest.mark.parametrize("symten_len", [4, 6])
class TestMechanics:
    """Test case for mechanical variables."""

    # TODO: use hypothesis testing here
    rng = np.random.default_rng()

    def generate_random_sig(self, symten_len: int):
        """Generate random stress matrix."""
        return self.rng.random((100000, symten_len)) * 1e6

    def test_frobenius_norm(self, symten_len: int):
        """Test Frobenius norm."""
        sig = self.generate_random_sig(symten_len)
        sig_mat = tensor_math.sym_tensor_to_mat(sig)
        frob2 = np.sqrt(
            tensor_math.trace(
                np.diagonal(np.transpose(sig_mat, (0, 2, 1)) @ sig_mat, 0, 2)
            )
        )
        assert_allclose(tensor_math.frobenius_norm(sig), frob2)

    def test_invariant_1(self, symten_len: int):
        """Test first invariant."""
        sig = self.generate_random_sig(symten_len)
        assert_allclose(
            tensor_math.invariant_1(sig),
            tensor_math.trace(tensor_math.eigenvalues(sig)),
        )

    def test_invariant_2(self, symten_len: int):
        """Test second invariant."""
        sig = self.generate_random_sig(symten_len)
        eig_vals = tensor_math.eigenvalues(sig)
        assert_allclose(
            tensor_math.invariant_2(sig),
            np.sum(eig_vals * np.roll(eig_vals, 1, axis=-1), axis=-1),
            rtol=5e-5,
        )

    def test_invariant_3(self, symten_len: int):
        """Test third invariant."""
        sig = self.generate_random_sig(symten_len)
        eig_vals = tensor_math.eigenvalues(sig)
        assert_allclose(
            tensor_math.invariant_3(sig), np.prod(eig_vals, axis=-1)
        )

    def test_deviator_invariant_1(self, symten_len: int):
        """Test first deviator invariant."""
        sig = self.generate_random_sig(symten_len)
        j1 = tensor_math.deviator_invariant_1(sig)
        assert_allclose(j1, np.zeros(j1.shape))

    def test_deviator_invariant_2(self, symten_len: int):
        """Test second deviator invariant."""
        sig = self.generate_random_sig(symten_len)
        mat_sig = tensor_math.sym_tensor_to_mat(sig)
        assert_allclose(
            tensor_math.deviator_invariant_2(sig),
            0.5
            * (
                tensor_math.matrix_trace(mat_sig @ mat_sig)
                - (1.0 / 3.0) * tensor_math.trace(sig) ** 2
            ),
        )

    def test_deviator_invariant_3(self, symten_len: int):
        """Test third deviator invariant."""
        sig = self.generate_random_sig(symten_len)
        mat_sig = tensor_math.sym_tensor_to_mat(sig)
        assert_allclose(
            tensor_math.deviator_invariant_3(sig),
            (1.0 / 3.0)
            * (
                tensor_math.matrix_trace(mat_sig @ mat_sig @ mat_sig)
                - tensor_math.matrix_trace(mat_sig @ mat_sig)
                * tensor_math.trace(sig)
                + (2.0 / 9.0) * tensor_math.trace(sig) ** 3.0
            ),
        )

    def test_von_mises(self, symten_len: int):
        """Test von Mises invariant."""
        sig = self.generate_random_sig(symten_len)
        ev = tensor_math.eigenvalues(sig)
        assert_allclose(
            tensor_math.von_mises(sig),
            np.sqrt(0.5 * np.sum(np.square(ev - np.roll(ev, 1, -1)), -1)),
        )

    def test_octahedral_shear_stress(self, symten_len: int):
        """Test octahedral shear stress invariant."""
        sig = self.generate_random_sig(symten_len)
        assert_allclose(
            tensor_math.octahedral_shear(sig),
            (1.0 / 3.0)
            * np.sqrt(
                2 * tensor_math.invariant_1(sig) ** 2
                - 6 * tensor_math.invariant_2(sig)
            ),
        )
