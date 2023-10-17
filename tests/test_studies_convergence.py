"""Unit tests for meshplotlib."""

import unittest

import matplotlib.pyplot as plt
import pandas as pd

from ogstools.propertylib.defaults import pressure
from ogstools.studies.convergence import (
    convergence,
    plot_convergence,
    richardson_extrapolation,
)
from ogstools.studies.examples import analytical_solution, meshes


class ConvergenceTest(unittest.TestCase):
    """Test case for meshplotlib."""

    def test_square_neumann_benchmark(self):
        target_mesh = meshes[0]
        richardson = richardson_extrapolation(meshes[-2], meshes[-1], pressure)
        analytical = analytical_solution(target_mesh)
        conv = convergence(target_mesh, meshes, analytical, pressure)
        pressure_expected = [
            0.030989251445358995,
            0.004311496211867697,
            0.0010968159126094655,
            0.00026568946729398736,
            6.283264310009035e-05,
            1.2467294614441292e-05,
        ]
        self.assertAlmostEqual(conv["pressure"], pressure_expected, places=5)
        pd.DataFrame(conv)
        _, axs = plt.subplots(dpi=200, figsize=[5, 3], nrows=1)
        _ = plot_convergence(target_mesh, meshes, richardson, pressure, axs)


if __name__ == "__main__":
    unittest.main(argv=[""], verbosity=2, exit=False)
