"""Unit tests for meshlib."""

import matplotlib.pyplot as plt
import pytest

import ogstools as ot
from ogstools import examples


@pytest.fixture()
def failing_model() -> ot.Model:

    msh_file = ot.gmsh_tools.cuboid(lengths=1.0, n_edge_cells=1, n_layers=1)
    meshes = ot.Meshes.from_gmsh(msh_file, dim=[1, 3], log=False)
    prj = ot.Project(input_file=examples.prj_aniso_expansion).copy()
    return ot.Model(prj, meshes, id="failing_model")


@pytest.fixture()
def good_model() -> ot.Model:
    return examples.load_model_liquid_flow_simple().copy(id="good_model")


@pytest.fixture(params=["failing_model", "good_model"])
def model(request: pytest.FixtureRequest) -> ot.Model:
    return request.getfixturevalue(request.param)


@pytest.mark.system()
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
def test_parallel_runs():
    """Simulations can run in parallel (native) or sequentially (interactive)."""
    model = examples.load_model_liquid_flow_simple()

    sim_c1 = model.controller()
    sim_c2 = model.controller()
    assert sim_c1.status == sim_c1.Status.running
    assert sim_c2.status == sim_c2.Status.running

    sims = [simc.run() for simc in [sim_c1, sim_c2]]
    assert sims[0].status == sim_c1.Status.done
    assert sims[1].status == sim_c2.Status.done
    assert sims[0].result == sims[1].result


@pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 30})
def test_simulation_log_convergence() -> plt.Figure:
    model = examples.load_model_liquid_flow_simple()
    sim = model.run()
    return sim.log.plot_convergence()


@pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 30})
def test_simulation_log_convergence_order() -> plt.figure:
    model = examples.load_model_liquid_flow_simple()
    sim = model.run()
    return sim.log.plot_convergence_order()
