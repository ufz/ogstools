import copy
from collections.abc import Callable
from typing import Any

import pytest

# Untypical import, technical necessary here because of specific approach to testing
# (one tests for multiple classes)
from ogstools import (
    Execution,
    Meshes,
    MeshSeries,
    Model,
    OGSInteractiveController,
    Project,
    Result,
    Simulation,
    StorageBase,
)
from ogstools.examples import (
    EXAMPLES_DIR,
    load_meshes_simple_lf,
    load_meshseries_CT_2D_XDMF,
    load_model_liquid_flow_simple,
    load_project_simple_lf,
    load_simulation_smalldeformation,
)

EVAL_NAMESPACE = {
    "Project": Project,
    "Model": Model,
    "Meshes": Meshes,
    "Simulation": Simulation,
    "Execution": Execution,
    "MeshSeries": MeshSeries,
    "Result": Result,
    "OGSInteractiveController": OGSInteractiveController,
}


def model_liquid_flow_simple_prjfile() -> Model:
    prj_file = load_simulation_smalldeformation().model.project.input_file
    return Model(project=prj_file)


def simulation_run_liquid_flow_simple_folder() -> Simulation:
    model_folder = EXAMPLES_DIR / "simulation" / "small_deformation" / "model"
    m = Model.from_folder(model_folder)
    return m.run()


def simulation_run() -> Simulation:
    model = model_liquid_flow_simple_prjfile()
    return model.run()


def test_framework_prj(tmp_path):
    from ogstools.examples import load_project_simple_lf

    StorageBase.Userpath = tmp_path / "framework_prj"

    model = load_project_simple_lf()
    assert not model.user_specified_target
    model.id = "my_project"
    model.__str__()
    model.__repr__()

    model_deep_copied = copy.deepcopy(model)
    assert model == model_deep_copied
    model.replace_text(4, ".//integration_order")
    assert model != model_deep_copied
    assert model is not model_deep_copied
    assert model.next_target != model_deep_copied.next_target

    # Reconstruction from repr only work when model was saved!
    model.save(overwrite=True)
    first_line = repr(model).split("\n", 1)[0]
    model_from_repr: Project = eval(first_line, EVAL_NAMESPACE)

    assert model == model_from_repr


@pytest.mark.tools  # NodeReordering
def test_framework_model():
    from ogstools.examples import load_model_liquid_flow_simple

    model = load_model_liquid_flow_simple()
    model.id = "my_model"
    model.__str__()
    model.__repr__()

    model_deep_copied = copy.deepcopy(model)
    assert model == model_deep_copied
    model.execution.interactive = not model.execution.interactive
    assert model != model_deep_copied
    assert model is not model_deep_copied
    assert model.next_target != model_deep_copied.next_target

    # Reconstruction from repr only work when model was saved
    model.save(overwrite=True)
    first_line = repr(model).split("\n", 1)[0]
    model_from_repr: Model = eval(first_line)

    assert model.meshes == model_from_repr.meshes
    assert model.execution == model_from_repr.execution

    assert model == model_from_repr


@pytest.mark.tools  # NodeReordering
def test_framework_meshes(tmp_path):
    from ogstools.examples import load_meshes_simple_lf

    StorageBase.Userpath = tmp_path / "test_framework_meshes"

    meshes_1 = load_meshes_simple_lf()
    meshes_1.id = "meshes_1"
    meshes_1.__str__()
    meshes_1.__repr__()

    meshes_1_deep_copied = copy.deepcopy(meshes_1)
    assert meshes_1 == meshes_1_deep_copied
    meshes_1.domain_name = meshes_1.domain_name + "_other"
    assert meshes_1 != meshes_1_deep_copied
    assert meshes_1 is not meshes_1_deep_copied
    assert meshes_1.next_target != meshes_1_deep_copied.next_target

    # Reconstruction from repr only work when model was saved
    meshes_1.save(overwrite=True)
    first_line = repr(meshes_1).split("\n", 1)[0]
    meshes_from_repr: Meshes = eval(first_line)
    assert meshes_1 == meshes_from_repr


@pytest.mark.tools  # NodeReordering
@pytest.mark.parametrize(
    "interactive",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.xdist_group("interactive_serial"),
        ),
    ],
    ids=["native", "interactive"],
)
def test_framework_simulation(interactive):
    """
    Check if Simulation object can be reconstructed
    Only smoke tests, since the controller has by design no __deepcopy__ and __eq__
    """
    model = load_model_liquid_flow_simple()
    model.execution.interactive = interactive
    sim = model.run()

    _str = sim.__str__()
    assert len(_str) > 50
    _repr = sim.__repr__()
    assert len(_repr) > 50

    first_line = repr(sim).split("\n", 1)[0]
    sim_from_repr: Simulation = eval(first_line, EVAL_NAMESPACE)

    assert sim_from_repr.log_file


@pytest.mark.tools  # NodeReordering
@pytest.mark.parametrize(
    "interactive",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.xdist_group("interactive_serial"),
        ),
    ],
    ids=["native", "interactive"],
)
def test_framework_simulation_controller(interactive):
    """
    Check if Simulation object can be reconstructed
    Only smoke tests, since the controller has by design no __deepcopy__ and __eq__
    """
    model = load_model_liquid_flow_simple()
    model.execution.interactive = interactive
    sim = model.controller()

    _str = sim.__str__()
    assert len(_str) > 50
    _repr = sim.__repr__()
    assert len(_repr) > 50

    # deepcopy not allowed
    # save not implemented
    first_line = repr(sim).split("\n", 1)[0]
    sim.run()
    sim_from_repr: OGSInteractiveController = eval(first_line, EVAL_NAMESPACE)

    if interactive:
        sim_from_repr.execute_time_step()
    else:
        sim_from_repr.terminate()  # no other action possible


def assert_framework_object_contract(
    *,
    factory: Callable[[], Any],
    mutate: Callable[[Any], None],
):
    """
    Generic contract test for StorageBase-derived framework objects.
    """

    obj = factory()
    assert not obj.user_specified_target
    # __str__ / __repr__ should not crash
    obj.__str__()
    obj.__repr__()

    obj_copy = copy.deepcopy(obj)
    assert obj_copy == obj
    assert obj is not obj_copy, "deepcopy changes identity"
    assert (
        obj.next_target != obj_copy.next_target
    ), "deepcopy changes identity (also in file)"

    if mutate:
        mutate(obj)
        assert obj != obj_copy, "mutation must break equality"

    # save + repr reconstruction
    obj.save(overwrite=True)
    first_line = repr(obj).split("\n", 1)[0]
    reconstructed = eval(first_line)
    assert obj == reconstructed
    if mutate:
        assert obj_copy != reconstructed


@pytest.mark.parametrize(
    ("factory", "mutate"),
    [
        # 0 Meshes
        pytest.param(
            load_meshes_simple_lf,
            lambda m: setattr(m, "domain_name", m.domain_name + "_other"),
            marks=pytest.mark.tools(),
            id="Meshes",
        ),
        # 1 Project
        pytest.param(
            load_project_simple_lf,
            lambda p: p.replace_text(4, ".//integration_order"),
            id="Project",
        ),
        # 2 Execution
        pytest.param(
            lambda: load_model_liquid_flow_simple().execution,
            lambda m: setattr(m, "interactive", not m.interactive),
            marks=pytest.mark.tools(),
            id="Execution",
        ),
        # 3 Model (objects)
        pytest.param(
            load_model_liquid_flow_simple,
            lambda m: setattr(
                m.execution, "interactive", not m.execution.interactive
            ),
            marks=pytest.mark.tools(),
            id="Model(objects)",
        ),
        # 4 Model (prj_file)
        pytest.param(
            model_liquid_flow_simple_prjfile,
            lambda m: setattr(
                m.execution, "interactive", not m.execution.interactive
            ),
            id="Model(prj-file)",
        ),
        # 5 Model (from folder)
        # Issue #3589
        # pytest.param(
        #    lambda: simulation_run_liquid_flow_simple_folder(),
        #    None,
        #    id="Model(folder)",
        # ),
        # 6 Simulation
        # immutable or at least intended as snapshot
        pytest.param(
            load_simulation_smalldeformation,
            None,
            id="Simulation",
        ),
        # 7 Simulation run (immutable or at least intended as snapshot)
        pytest.param(
            simulation_run,
            None,
            marks=pytest.mark.system(),
            id="Simulation(run)",
        ),
        # 8 MeshSeries
        pytest.param(
            load_meshseries_CT_2D_XDMF,
            # lambda ms: ms.scale("km"),
            lambda ms: ms.extend(ms),
            id="MeshSeries",
        ),
    ],
)
def test_framework_objects(tmp_path, factory, mutate):
    StorageBase.Userpath = tmp_path / "test_framework_objects"
    assert_framework_object_contract(factory=factory, mutate=mutate)


@pytest.mark.parametrize(
    ("cls_name", "factory"),
    [
        pytest.param("Project", load_project_simple_lf, id="Project"),
        pytest.param(
            "Execution", lambda: Execution(mpi_ranks=4), id="Execution"
        ),
        pytest.param(
            "Meshes",
            load_meshes_simple_lf,
            marks=pytest.mark.tools(),
            id="Meshes",
        ),
        pytest.param(
            "Model",
            load_model_liquid_flow_simple,
            marks=pytest.mark.tools(),
            id="Model",
        ),
        pytest.param("MeshSeries", load_meshseries_CT_2D_XDMF, id="MeshSeries"),
        pytest.param(
            "Simulation", load_simulation_smalldeformation, id="Simulation"
        ),
    ],
)
def test_from_id_roundtrip(tmp_path, cls_name, factory):
    """
    Test that objects can be saved with an ID and reloaded using from_id().
    Also verifies that __repr__() shows from_id() and can be used to reconstruct.
    """
    StorageBase.Userpath = tmp_path / "from_id_roundtrip"
    # Create object and make a copy to avoid overwrite issues
    obj_original = factory()
    test_id = f"test_{cls_name.lower()}_id"
    obj = obj_original.copy(id=test_id)
    obj.save(overwrite=True)

    first_line = repr(obj).split("\n", 1)[0]
    assert "test_" in first_line, f"Expected test_ in repr: {first_line}"
    assert (
        test_id in first_line
    ), f"Expected ID '{test_id}' in repr: {first_line}"

    obj_from_repr = eval(first_line, EVAL_NAMESPACE)

    assert (
        obj == obj_from_repr
    ), f"{cls_name} repr reconstruction should create equal object"
    assert obj_from_repr.id == test_id, "Loaded object has same id."
    copied = obj_from_repr.copy(id=test_id)
    assert copied.id == test_id, "Reassigned id"
