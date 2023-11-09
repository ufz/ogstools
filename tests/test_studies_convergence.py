"""Unit tests for meshplotlib."""

import unittest

import numpy as np

from ogstools.propertylib import Scalar
from ogstools.studies.convergence import (
    convergence_metrics,
    grid_convergence,
    log_fit,
    plot_convergence,
    plot_convergence_errors,
    richardson_extrapolation,
)
from ogstools.studies.convergence.examples import analytical_solution, meshes


class ConvergenceTest(unittest.TestCase):
    """Test case for convergent meshes."""

    def test_square_neumann_benchmark(self):
        topology = meshes[-3]
        mesh_property = Scalar("pressure", "m", "m")
        conv = grid_convergence(
            meshes, mesh_property, topology, refinement_ratio=2.0
        )
        richardson = richardson_extrapolation(meshes, mesh_property, topology)
        np.testing.assert_allclose(conv["r"], richardson["r"], rtol=1e-10)
        analytical = analytical_solution(topology)
        np.testing.assert_allclose(
            richardson[mesh_property.data_name],
            analytical[mesh_property.data_name],
            rtol=2e-3,
            verbose=True,
        )
        metrics = convergence_metrics(meshes, richardson, mesh_property)
        el_len = metrics["mean element length"].to_numpy()[:-1]
        re_max = metrics["rel. error (max)"].to_numpy()[:-1]
        ratio = re_max[:-1] / re_max[1:]
        np.testing.assert_array_less(np.ones(ratio.shape) * 2.0, ratio)

        order, _ = log_fit(el_len, re_max)
        self.assertGreater(order, 2.0)

        _ = plot_convergence(metrics, mesh_property)
        _ = plot_convergence_errors(metrics)


if __name__ == "__main__":
    unittest.main(argv=[""], verbosity=2, exit=False)
