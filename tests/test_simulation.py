"""Unit tests for meshlib."""

import os
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

import ogstools as ot
from ogstools import examples


@pytest.fixture
def failing_model() -> ot.Model:
    msh_file = ot.gmsh_tools.cuboid(lengths=1.0, n_edge_cells=1, n_layers=1)
    meshes = ot.Meshes.from_gmsh(msh_file, dim=[1, 3], log=False)
    prj = ot.Project(input_file=examples.prj_aniso_expansion).copy()
    model = ot.Model(prj, meshes)
    model.id = "failing_model"
    return model


@pytest.fixture
def good_model() -> ot.Model:
    return examples.load_model_liquid_flow_simple().copy(id="good_model")


@pytest.fixture(params=["failing_model", "good_model"])
def model(request: pytest.FixtureRequest) -> ot.Model:
    return request.getfixturevalue(request.param)


@pytest.mark.system
def test_simulation_simple(tmp_path, good_model):
    sim = good_model.copy().run()
    assert sim.status == sim.Status.done
    sim_out = tmp_path / "sim_good_model"
    sim.save(sim_out)
    ms = sim.meshseries
    assert ms[-1]
    assert (sim_out / "model").is_symlink()
    assert (sim_out / "result").is_symlink()


@pytest.mark.system
def test_simulation_simple2(tmp_path, good_model):
    sim_out = tmp_path / "Simulation" / "sim_good_model"
    model = good_model.copy()
    sim = model.run(sim_out)
    assert sim.status == sim.Status.done
    assert sim.meshseries
    assert not (sim_out / "model").is_symlink()
    assert (sim_out / "model").is_dir()
    sim.save(tmp_path / "Simulation" / "model_save_as")


@pytest.mark.system
def test_simulation_simple_archive(tmp_path, good_model):
    sim_out = tmp_path / "Simulation" / "sim_good_model"
    model = good_model.copy()
    model.save(tmp_path / "model", archive=True)
    sim = model.run(sim_out)
    assert sim.meshseries
    assert (sim_out / "model").is_symlink()
    assert (sim_out / "model").is_dir()


@pytest.mark.system
@pytest.mark.skipif(
    (os.cpu_count() or 0) < 3 or sys.platform != "linux",
    reason="requires at least 3 CPUs and Linux",
)
@pytest.mark.usefixtures("require_ogs_containers")
@pytest.mark.parametrize("n", [1, 2, 3])
def test_simulation_parallel(good_model, n):
    parallel_model = good_model.copy()
    parallel_model.execution.omp_num_threads = 1  # no over-subscription
    parallel_model.execution.mpi_ranks = n
    parallel_model.execution.ogs = ot.Execution.CONTAINER_PARALLEL
    parallel_model.execution.log_level = "debug"
    parallel_model.execution.args = "--log-parallel"
    parallel_model.save()
    cmd = parallel_model.cmd

    # test
    mpi_part = rf"mpirun -np {n} " if n >= 1 else ""
    mesh_suffix = rf"/partition/{n}" if n > 1 else ""
    assert re.search(rf"apptainer exec \S+ {mpi_part}ogs", cmd)
    assert re.search(rf"-m \S+/meshes{mesh_suffix}", cmd)
    assert re.search(r"\S+/project/default\.prj", cmd)
    assert "-l debug" in cmd
    sim = parallel_model.run()
    assert len(sim.meshseries) > 2
    log_content = sim.log_file.read_text()
    assert f"with MPI. MPI processes: {n}." in log_content

    assert sim.status == sim.Status.done


@pytest.mark.system
@pytest.mark.xfail(
    sys.platform == "darwin",
    reason="OGS accidentally compiled without OpenMP on Mac in dependent ogs wheel package",
)
def test_simulation_omp_num_threads(good_model):
    model_with_threads = good_model.copy()
    model_with_threads.execution.omp_num_threads = 2
    sim = model_with_threads.run()
    log_content = sim.log_file.read_text()
    assert "OMP_NUM_THREADS is set to: 2" in log_content

    sim_no_threads = good_model.run()  # default has num_threads not set
    log_no_threads = sim_no_threads.log_file.read_text()
    assert good_model.execution.omp_num_threads is None
    assert "OMP_NUM_THREADS is not set" in log_no_threads


@pytest.mark.system
@pytest.mark.usefixtures(
    "require_ogs_wheel"
)  # only here we are sure that build with openmp
def test_simulation_ogs_asm_threads():
    model = examples.load_simulation_smalldeformation().model.copy(
        id="asm_test"
    )
    model_with_asm = model.copy()
    model_with_asm.execution.ogs_asm_threads = 2
    sim = model_with_asm.run()
    log_content = sim.log_file.read_text()
    assert "Threads used for ParallelVectorMatrixAssembler: 2" in log_content

    sim_no_asm = model.run()
    log_no_asm = sim_no_asm.log_file.read_text()
    # if not set (OGS intern), default is 1
    assert "Threads used for ParallelVectorMatrixAssembler: 1" in log_no_asm


@pytest.mark.system
@pytest.mark.skipif(sys.platform != "linux", reason="Linux only")
@pytest.mark.usefixtures("require_ogs_containers")
def test_simulation_container(good_model):
    parallel_model = good_model.copy()
    parallel_model.execution.omp_num_threads = 1
    parallel_model.execution.mpi_ranks = 2
    parallel_model.execution.ogs = ot.Execution.CONTAINER_PARALLEL
    parallel_model.execution.log_level = "debug"
    parallel_model.save()
    cmd = parallel_model.cmd
    assert re.search(r"apptainer exec \S+ mpirun -np 2 \S*ogs", cmd)
    assert re.search(r"-m \S+/meshes/partition/2", cmd)
    assert re.search(r"\S+/project/default\.prj", cmd)
    assert "-l debug" in cmd


@pytest.mark.system
@pytest.mark.parametrize("do_kill", [False, True], ids=["no-kill", "kill"])
@pytest.mark.parametrize(
    "interactive", [False], ids=["native"]
)  # ToDo: Issue #3589 - Interactive not yet working in (scheduler) parallel
def test_abort_run_and_status(
    model: ot.Model, do_kill: bool, interactive: bool
) -> None:
    """
    Test the normal model.run() but without directly calling
    """

    model.execution.interactive = interactive

    # run the model till the end
    sim_control = model.controller()  # always running in background
    # never set a break point between start and here (test is sensible to timing)
    assert sim_control.status == sim_control.Status.running
    d = sim_control.status_str()
    assert "running" in d
    if model._id == "failing_model":
        expected_final_state = sim_control.Status.error
    elif model._id == "good_model":
        expected_final_state = sim_control.Status.done

    else:
        expected_final_state = sim_control.Status.done

    if do_kill:
        sim_control.terminate()
        assert sim_control.status != sim_control.status.running
        assert "running" not in sim_control.status_str()
    else:
        # it takes a while after
        assert (
            sim_control.status == sim_control.Status.running
            or sim_control.status == expected_final_state
        )

        sim_control.run()
        assert sim_control.status == expected_final_state

        match expected_final_state:
            case sim_control.Status.error:
                assert "error" in sim_control.status_str()
            case sim_control.Status.done:
                assert "successfully" in sim_control.status_str()
            case _:
                pytest.fail(f"Unhandled final state: {expected_final_state}")


# ToDo: Issue #3589 + Console capture not thread safe - Test interactive
@pytest.mark.tools  # NodeReordering
def test_parallel_runs():
    """Simulations can run in parallel (native) or sequentially (interactive)."""
    model = examples.load_model_liquid_flow_simple()

    sim_c1 = model.controller()
    sim_c2 = model.controller()
    assert sim_c1.status in (sim_c1.Status.running, sim_c1.Status.done)
    assert sim_c2.status in (sim_c2.Status.running, sim_c2.Status.done)

    sims = [simc.run() for simc in [sim_c1, sim_c2]]
    assert sims[0].status == sim_c1.Status.done
    assert sims[1].status == sim_c2.Status.done
    assert sims[0].meshseries == sims[1].meshseries


@pytest.mark.system
def test_simulation_cmd_reproduces_result(tmp_path, good_model):
    """Run a simulation, save as archive, delete original, re-run via cmd."""
    import shutil

    sim = good_model.copy().run()
    sim.save(tmp_path / "sim_original", archive=True)

    # Delete where the first simulation was computed so cmd can reuse the path
    original_result_path = sim.result.next_target
    shutil.rmtree(original_result_path)
    original_result_path.mkdir()

    ret = subprocess.run(
        sim.cmd, shell=True, capture_output=True, text=True, check=False
    )
    assert ret.returncode == 0, ret.stderr

    assert ret.stdout != sim.log_file.read_text()
    ms_rerun = ot.MeshSeries(
        original_result_path / sim.model.project.meshseries_file()
    )
    assert sim.meshseries == ms_rerun


@pytest.mark.tools  # NodeReordering
@pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 30})
def test_plot_simulation_log_convergence() -> plt.Figure:
    model = examples.load_model_liquid_flow_simple()
    sim = model.run()
    return sim.log.plot_convergence()


@pytest.mark.tools  # NodeReordering
@pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 30})
def test_plot_simulation_log_convergence_order() -> plt.figure:
    model = examples.load_model_liquid_flow_simple()
    sim = model.run()
    return sim.log.plot_convergence_order()


def test_mock_model_restart() -> None:
    sim_dir = examples.EXAMPLES_DIR / "simulation/restart/sim"
    sim = ot.Simulation.from_folder(sim_dir)
    prj_ref = ot.Project(sim_dir / "model/project/ref.prj")
    new_timevalues = [2, 3, 4]
    model_restart = sim.restart(timevalues=new_timevalues)
    model_restart.save()

    assert prj_ref == model_restart.project


@pytest.mark.tools  # NodeReordering
@pytest.mark.usefixtures("require_ogs_containers")
def test_execution_defaults_from_env(monkeypatch, good_model):
    """OGS_EXECUTION_DEFAULTS env var loads settings from the example YAML."""
    yml_path = Path(ot.__file__).parent / "core/execution_default_example.yml"
    monkeypatch.setenv("OGS_EXECUTION_DEFAULTS", str(yml_path))
    exec_defaults = ot.Execution.from_default()
    assert exec_defaults.mpi_ranks == 2
    good_model.execution = exec_defaults
    sim = good_model.run()
    assert sim.status == sim.Status.done


def test_restart_error_cases():
    sim = ot.Simulation.from_folder(
        examples.EXAMPLES_DIR / "simulation/restart/sim"
    )
    # TypeError if both timevalues and (t_initial, t_end) are given
    with pytest.raises(TypeError):
        sim.restart(
            timevalues=[1, 2, 3, 4],
            t_initial=1,
            t_end=4,
            initial_dt=1,
        )
    # AssertionError if the input timevalues is []
    with pytest.raises(AssertionError):
        sim.restart(
            timevalues=[],
        )
    # AssertionError if the timevalues are not sorted
    with pytest.raises(AssertionError):
        sim.restart(
            timevalues=[4, 5, 2, 1],
        )
    return


@pytest.mark.system
def test_model_restart(tmp_path) -> None:
    model = ot.examples.load_model_liquid_flow_simple().copy()
    sim = model.run()
    assert sim.status == sim.Status.done
    sim.save(tmp_path / "entire_run")

    ms_full = sim.meshseries
    timestep_break = int(ms_full.timevalues.size / 2)
    new_timevalues = ms_full.timevalues[timestep_break:]

    model_restart = sim.restart(timevalues=new_timevalues)
    sim_restart = model_restart.run()

    ms_restart = sim_restart.meshseries
    assert all(ms_restart.timevalues == new_timevalues)
    assert ot.MeshSeries.compare(ms_full[timestep_break:], ms_restart)


@pytest.mark.system
def test_simulation_status_invalid_prj() -> None:

    file = (
        examples.EXAMPLES_DIR
        / "prj"
        / "Elliptic/quarter_circle/quarter_circle_nodal_source_term.prj"
    )
    prj = ot.Project(input_file=file).copy()
    prj.add_element(
        tag="invalid", attrib_list=["invalid"], attrib_value_list=["0"]
    )
    model = ot.Model(prj)
    sim = model.run()
    assert sim.status == sim.Status.error
