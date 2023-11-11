"""Unit tests for meshplotlib."""

import unittest
from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp

import numpy as np
import pyvista as pv
from ogs6py import ogs

from ogstools import meshlib, msh2vtu, propertylib
from ogstools.studies import convergence
from ogstools.studies.convergence.examples import (
    steady_state_diffusion_analytical_solution,
)


class ConvergenceTest(unittest.TestCase):
    """Test case for convergent meshes."""

    def test_square_neumann_benchmark(self):
        temp_dir = Path(mkdtemp())
        sim_results = []
        for i in range(3, 6):
            msh_path = temp_dir / "square.msh"
            meshlib.gmsh_meshing.rect_mesh(
                n_edge_cells=2**i,
                structured_grid=True,
                out_name=msh_path,
            )
            msh2vtu.msh2vtu(input_filename=msh_path, output_path=temp_dir)
            model = ogs.OGS(
                PROJECT_FILE=temp_dir / "default.prj",
                INPUT_FILE=convergence.examples.steady_state_diffusion_prj,
            )
            model.write_input()
            ogs_args = f"-m {temp_dir} -o {temp_dir}"
            model.run_model(write_logs=False, args=ogs_args)

            result = meshlib.MeshSeries(
                str(temp_dir / "steady_state_diffusion.pvd")
            ).read(-1)
            result_path = temp_dir / f"steady_state_diffusion_{i}.vtu"
            result.save(result_path)
            sim_results += [pv.read(result_path)]

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
            sim_results, richardson, mesh_property
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
