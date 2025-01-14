"""Unit tests for convergence studies."""

from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp

import numpy as np
import pyvista as pv

import ogstools as ot
from ogstools.examples import analytical_diffusion, prj_steady_state_diffusion
from ogstools.studies import convergence


class TestConvergence:
    """Test case for convergent meshes."""

    def test_steady_state_diffusion(self):
        temp_dir = Path(mkdtemp(prefix="test_steady_state_diffusion"))
        sim_results = []
        edge_cells = [2**i for i in range(3, 6)]
        for n_edge_cells in edge_cells:
            msh_path = temp_dir / "square.msh"
            ot.meshlib.gmsh_meshing.rect(
                n_edge_cells=n_edge_cells,
                structured_grid=True,
                out_name=msh_path,
            )
            meshes = ot.meshes_from_gmsh(filename=msh_path, log=False)
            for name, mesh in meshes.items():
                pv.save_meshio(Path(temp_dir, name + ".vtu"), mesh)
            model = ot.Project(
                output_file=temp_dir / "default.prj",
                input_file=prj_steady_state_diffusion,
            )
            prefix = "steady_state_diffusion_" + str(n_edge_cells)
            model.replace_text(prefix, ".//prefix")
            model.write_input()
            ogs_args = f"-m {temp_dir} -o {temp_dir}"
            model.run_model(write_logs=False, args=ogs_args)
            sim_results += [
                ot.MeshSeries(str(temp_dir / (prefix + ".pvd"))).mesh(-1)
            ]

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
        analytical = analytical_diffusion(topology)
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

        rmtree(temp_dir)
