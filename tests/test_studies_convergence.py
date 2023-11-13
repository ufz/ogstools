"""Unit tests for meshplotlib."""

import unittest
from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp

import numpy as np
from ogs6py import ogs

from ogstools import meshlib, msh2vtu, propertylib
from ogstools.studies import convergence
from ogstools.studies.convergence.examples import (
    steady_state_diffusion_analytical_solution,
)


class ConvergenceTest(unittest.TestCase):
    """Test case for convergent meshes."""

    def test_steady_state_diffusion(self):
        temp_dir = Path(mkdtemp(prefix="test_steady_state_diffusion"))
        sim_results = []
        edge_cells = [2**i for i in range(3, 6)]
        for n_edge_cells in edge_cells:
            msh_path = temp_dir / "square.msh"
            meshlib.gmsh_meshing.rect(
                n_edge_cells=n_edge_cells,
                structured_grid=True,
                out_name=msh_path,
            )
            msh2vtu.msh2vtu(
                input_filename=msh_path, output_path=temp_dir, log_level="ERROR"
            )
            model = ogs.OGS(
                PROJECT_FILE=temp_dir / "default.prj",
                INPUT_FILE=convergence.examples.steady_state_diffusion_prj,
            )
            prefix = "steady_state_diffusion_" + str(n_edge_cells)
            model.replace_text(prefix, ".//prefix")
            model.write_input()
            ogs_args = f"-m {temp_dir} -o {temp_dir}"
            model.run_model(write_logs=False, args=ogs_args)
            sim_results += [
                meshlib.MeshSeries(str(temp_dir / (prefix + ".pvd"))).read(-1)
            ]

        topology = sim_results[-3]
        spacing = convergence.add_grid_spacing(topology)["grid_spacing"]
        np.testing.assert_array_less(0.0, spacing)
        mesh_property = propertylib.Scalar("pressure", "m", "m")
        conv = convergence.grid_convergence(
            sim_results, mesh_property, topology, refinement_ratio=2.0
        )
        richardson = convergence.richardson_extrapolation(
            sim_results, mesh_property, topology, refinement_ratio=2.0
        )
        np.testing.assert_allclose(conv["r"], richardson["r"], rtol=1e-10)
        analytical = steady_state_diffusion_analytical_solution(topology)
        np.testing.assert_allclose(
            richardson[mesh_property.data_name],
            analytical[mesh_property.data_name],
            rtol=2e-3,
            verbose=True,
        )
        metrics = convergence.convergence_metrics(
            sim_results, richardson, mesh_property, []
        )
        el_len = metrics["mean element length"].to_numpy()[:-1]
        re_max = metrics["rel. error (max)"].to_numpy()[:-1]
        ratio = re_max[:-1] / re_max[1:]
        np.testing.assert_array_less(2.0, ratio)

        order, _ = convergence.log_fit(el_len, re_max)
        self.assertGreater(order, 2.0)

        _ = convergence.plot_convergence(metrics, mesh_property)
        _ = convergence.plot_convergence_errors(metrics)

        rmtree(temp_dir)
