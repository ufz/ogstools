"""Unit tests for convergence studies."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

import ogstools as ot
from ogstools.examples import anasol, prj_steady_state_diffusion
from ogstools.studies import convergence


class TestConvergence:
    """Test case for convergent meshes."""

    @pytest.mark.system()
    def test_steady_state_diffusion(self):
        sim_results = []
        edge_cells = [2**i for i in range(3, 6)]
        simulations = []
        for n_edge_cells in edge_cells:
            meshes = ot.Meshes.from_gmsh(
                ot.gmsh_tools.rect(
                    n_edge_cells=n_edge_cells, structured_grid=True
                )
            )
            prj = ot.Project(input_file=prj_steady_state_diffusion).copy()
            prefix = "steady_state_diffusion_" + str(n_edge_cells)
            prj.replace_text(prefix, ".//prefix")
            model = ot.Model(prj, meshes)
            sim_c = model.controller()
            simulations.append((prefix, sim_c))

        sim_results = [sc[1].run().result[-1] for sc in simulations]
        topology = sim_results[-3]
        spacing = convergence.add_grid_spacing(topology)["grid_spacing"]
        np.testing.assert_array_less(0.0, spacing)
        variable = ot.variables.Scalar("pressure", "m", "m")
        conv = convergence.grid_convergence(
            sim_results, variable, topology, refinement_ratio=2.0
        )
        richardson = convergence.richardson_extrapolation(
            sim_results, variable, topology, refinement_ratio=2.0
        )
        np.testing.assert_allclose(conv["r"], richardson["r"], rtol=1e-10)
        analytical = anasol.diffusion_head_analytical(topology)
        np.testing.assert_allclose(
            richardson[variable.data_name],
            analytical[variable.data_name],
            rtol=2e-3,
            verbose=True,
        )
        metrics = convergence.convergence_metrics(
            sim_results, richardson, variable, []
        )
        el_len = metrics["mean element length"].to_numpy()[:-1]
        re_max = metrics["rel. error (max)"].to_numpy()[:-1]
        ratio = re_max[:-1] / re_max[1:]
        np.testing.assert_array_less(2.0, ratio)

        order, _ = convergence.log_fit(el_len, re_max)
        assert order > 2.0

        _ = convergence.plot_convergence(metrics, variable)
        _ = convergence.plot_convergence_errors(metrics)
        plt.close()
