from pathlib import Path
from typing import Any

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import HealthCheck, assume, settings
from hypothesis.stateful import (
    RuleBasedStateMachine,
    invariant,
    precondition,
    rule,
    run_state_machine_as_test,
)

import ogstools as ot
from ogstools.examples import (
    EXAMPLES_DIR,
    load_meshseries_CT_2D_XDMF,
    load_meshseries_HT_2D_VTU,
    load_model_liquid_flow_simple,
    load_simulation_smalldeformation,
    prj_mechanics,
)
from ogstools.gmsh_tools import rect

ot.StorageBase.Backup = False


@settings(
    max_examples=20,
    suppress_health_check=[HealthCheck.filter_too_much],
)
class ProjectMachine(RuleBasedStateMachine):
    temp: Path
    TestClass: Any
    file_counter = 0

    def __init__(self):
        super().__init__()
        self.storable = None  # current Project
        self.TestClass = ProjectMachine.TestClass
        self.tmp = ProjectMachine.temp

    @rule(
        full_path=st.one_of(st.just(True), st.just(None)),
        assign_id=st.booleans(),
    )
    @precondition(lambda self: self.storable is None)
    def create(self, full_path, assign_id):

        ProjectMachine.file_counter += 1
        if full_path is True:
            output_file = self.tmp / f"file_{ProjectMachine.file_counter}"
        ## Could only be tested locally when cwd is guaranteed to be clear
        # elif full_path is False:
        #    output_file = f"file_{ProjectMachine.file_counter}"
        else:
            output_file = None

        if self.TestClass == ot.Project:
            self.storable = self.TestClass(
                input_file=prj_mechanics, output_file=output_file
            ).copy()
        elif self.TestClass == ot.Meshes:
            self.storable = ot.Meshes.from_gmsh(rect())

        elif self.TestClass == ot.Model:
            self.storable = load_model_liquid_flow_simple()

        elif self.TestClass == ot.Simulation:
            self.storable = load_simulation_smalldeformation()

        elif self.TestClass == ot.MeshSeries:
            self.storable = load_meshseries_HT_2D_VTU()
        else:
            msg = "No specialization"
            raise ValueError(msg)

        if assign_id:
            self.storable.id = f"id_{ProjectMachine.file_counter}"
            assert self.storable.user_specified_target

    @rule(
        filename=st.one_of(st.booleans(), st.just(None)),
        overwrite=st.booleans(),
        archive=st.booleans(),
    )
    @precondition(
        lambda self: self.storable is not None
        and (
            self.storable.active_target is None
            or not self.storable.active_target.exists()
        )
        and (not self.storable.user_specified_target)
    )
    def save_new(self, filename, archive, overwrite):
        """
        Save for the first time, no problem independent from overwrite
        """
        assume(not self.storable.user_specified_target)
        assume((self.storable.user_specified_target or filename) and archive)
        ProjectMachine.file_counter += 1

        suffix = "." + self.storable._ext if self.storable._ext else ""

        if filename:
            filename = self.tmp / f"file_{ProjectMachine.file_counter}{suffix}"
            self.storable.save(filename, overwrite=overwrite, archive=archive)
        else:
            self.storable.save(overwrite=overwrite, archive=archive)

    @rule(filename=st.booleans())
    @precondition(lambda self: self.storable is not None)
    def save_overwrite(self, filename):
        if filename:
            suffix = "." + self.storable._ext if self.storable._ext else ""
            filename = self.tmp / f"file_{ProjectMachine.file_counter}{suffix}"
            self.storable.save(filename, overwrite=True)
            assert self.storable.user_specified_target
        else:
            self.storable.save(overwrite=True)

    @rule(overwrite=st.booleans())
    @precondition(
        lambda self: self.storable is not None and self.storable.is_saved
    )
    def save_again(self, overwrite):
        if overwrite or not self.storable.user_specified_target:
            self.storable.save(overwrite=overwrite)
        else:
            with pytest.raises(ValueError, match="overwrite"):
                self.storable.save(overwrite=overwrite)

    @rule(set_new=st.booleans())
    @precondition(
        lambda self: self.storable is not None and self.storable.is_saved
    )
    def load(self, set_new):
        # if we say it's saved, file must exist and be loadable
        assert self.storable.active_target is not None
        if ProjectMachine.TestClass == ot.Project:
            loaded = ot.Project.from_folder(self.storable.active_target)
            assert isinstance(loaded, ot.Project)
            if set_new:
                self.storable = loaded

        elif ProjectMachine.TestClass == ot.Meshes:
            loaded = ot.Meshes.from_folder(filepath=self.storable.active_target)
            assert isinstance(loaded, ot.Meshes)
            if set_new:
                self.storable = loaded

    @invariant()
    @precondition(lambda self: self.storable is not None)
    def saved_file_exists_if_marked(self):
        if self.storable.is_saved:
            assert self.storable.active_target is not None
            assert self.storable.active_target.exists()
        else:
            assert (
                not self.storable.active_target
                or not self.storable.active_target.exists()
            )


def assert_files_saved(
    files, expected_count=None, dry_run=False, allow_symlinks=False
):
    """Helper to verify that save() returns correct file paths.

    Args:
        files: List of file paths returned by save()
        expected_count: Expected number of files (None to skip count check)
        dry_run: If True, assert files do NOT exist; if False, assert they exist
        allow_symlinks: If True, also accept symlinks as existing
    """
    assert isinstance(files, list), "save() should return a list"
    assert len(files) > 0, "save() should return at least one file"
    if expected_count is not None:
        assert (
            len(files) == expected_count
        ), f"Expected {expected_count} files, got {len(files)}"
    for f in files:
        exists = f.exists() or (allow_symlinks and f.is_symlink())
        if dry_run:
            assert not exists, f"File should not exist in dry_run: {f}"
        else:
            assert exists, f"Returned file should exist: {f}"


class TestStorage:
    """Test case for ogstools utilities."""

    @pytest.mark.parametrize(
        "test_class",
        [ot.Project, ot.Meshes, ot.Model, ot.Simulation, ot.MeshSeries],
        ids=["Project", "Meshes", "Model", "Simulation", "MeshSeries"],
    )
    def test_state_machine(self, tmp_path, test_class):
        ot.StorageBase.Userpath = tmp_path
        m = ProjectMachine
        m.temp = tmp_path
        m.TestClass = test_class
        run_state_machine_as_test(m)

    def test_prj_save_load_roundtrip(self, tmp_path):

        prj1 = ot.Project(prj_mechanics).copy(tmp_path / "prj1")
        files1 = prj1.save()
        prj2 = ot.Project(input_file=files1[0]).copy(tmp_path / "prj2")
        files2 = prj2.save()
        assert files1 != files2
        assert prj1 == prj2

    def test_model_save_load_roundtrip(self):

        meshes = ot.Meshes.from_gmsh(rect())
        model1 = ot.Model(project=prj_mechanics, meshes=meshes)
        files1 = model1.save()
        model2 = ot.Model.from_folder(model1.active_target)
        with pytest.raises(ValueError, match="overwrite"):
            model2.save()
        files3 = model2.save(overwrite=True)
        assert files1 != files3
        assert model1 == model2

    def test_meshes_save_load_roundtrip(self):
        meshes1 = ot.Meshes.from_gmsh(rect())
        files1 = meshes1.save()
        assert meshes1.validate()
        meshes2 = ot.Meshes.from_folder(meshes1.active_target)
        with pytest.raises(ValueError, match="overwrite"):
            files2 = meshes2.save()

        files2 = meshes2.save(overwrite=True)
        assert meshes2.validate()
        assert files1 != files2
        assert meshes1 == meshes2

        meshes3 = ot.Meshes.from_gmsh(rect(lengths=11))
        assert meshes3 != meshes1

    def test_sim_save_load_roundtrip(self):
        # ToDo example ot.Simulation()
        prj1 = ot.Project(input_file=prj_mechanics).copy()
        meshes1 = ot.Meshes.from_gmsh(rect(n_edge_cells=12))
        m = ot.Model(prj1, meshes1)
        sim1 = m.run()

        files1 = sim1.save()
        sim2 = ot.Simulation.from_folder(sim1.active_target)
        files2 = sim2.save()
        if files1:
            assert files1 != files2
        assert sim1 == sim2

    def test_storage_project(self, tmp_path):
        # uses gml
        prj_file = (
            EXAMPLES_DIR
            / "prj"
            / "HydroMechanics/AnchorSourceTerm/two_anchors.prj"
        )
        prj = ot.Project(prj_file)
        files = prj.save(tmp_path / "test_storage")
        assert prj.active_target
        assert_files_saved(files)
        gml_file = prj.geometry.active_target
        assert gml_file
        assert gml_file.exists()

    def test_storage_project_python_script(self, tmp_path):
        """Test that python_script files are saved inside the project folder."""
        # uses python_script
        prj_file = EXAMPLES_DIR / "prj" / "nuclear_decay.prj"
        prj = ot.Project(prj_file)

        # Verify python_script is loaded correctly
        assert prj.python_script.filename == "decay_boundary_conditions.py"
        assert prj.python_script.active_target is not None
        assert prj.python_script.active_target.exists()

        target = tmp_path / "test_pyscript"
        files = prj.save(target)
        assert_files_saved(files, expected_count=2)
        assert prj.active_target == target

        py_file = prj.python_script.active_target
        assert py_file is not None
        assert py_file.exists()
        assert py_file.name == "decay_boundary_conditions.py"

        original_py = prj_file.parent / "decay_boundary_conditions.py"
        assert py_file.read_bytes() == original_py.read_bytes()

    def test_storage_model_1(self, tmp_path):
        prj1 = ot.Project(input_file=prj_mechanics, output_file="mechanics")
        meshes1 = ot.Meshes.from_gmsh(rect(n_edge_cells=12))
        files_dry = prj1.save(tmp_path / "vtk", overwrite=True, dry_run=True)
        assert_files_saved(files_dry, dry_run=True)
        files = prj1.save(tmp_path / "vtk", overwrite=True)
        model_1_1 = ot.Model(prj1, meshes1, id="model_1_1")
        files_overwrite = model_1_1.save("m", overwrite=True)
        assert_files_saved(files_overwrite)
        assert files_dry == files
        sim = model_1_1.run()
        sim.save(tmp_path / "y", overwrite=True, archive=True)
        ms = sim.result
        ms.save(tmp_path / "my_ms.pvd", overwrite=True)
        assert sim.result.filepath.exists()

    def test_storage_sim(self, tmp_path):
        model = load_simulation_smalldeformation()
        model.save(tmp_path / "mysim", overwrite=True, archive=True)
        assert model.user_specified_target
        assert model.active_target.exists()
        model.save(overwrite=True)
        assert model.active_target.exists()

    def test_storage_model(self, tmp_path):
        model = load_model_liquid_flow_simple()
        model.save(tmp_path / "mytest", overwrite=True)
        assert model.active_target.exists()
        model.save(overwrite=True)
        assert model.active_target.exists()

    def test_storage_meshes(self, tmp_path):
        meshes1 = ot.Meshes.from_gmsh(rect(n_edge_cells=12))
        meshes1.id = "meshes1"
        meshes1.save(tmp_path / "new_name", overwrite=True)
        meshes1.save(overwrite=True)

    def test_storage_meshseries(self, tmp_path):
        # ToDo smaller --> better, or limit samples
        meshseries1 = load_meshseries_CT_2D_XDMF()
        meshseries1.id = "meshseries1"
        meshseries1.save(tmp_path / "new_meshseries.pvd", overwrite=True)
        meshseries1.save(overwrite=True)

    @pytest.mark.parametrize(
        "save_strategy",
        ["no", "id", "target", "empty"],
    )
    def test_storage_multi_model_multi_sim(self, tmp_path, save_strategy):
        ot.StorageBase.Userpath = tmp_path
        prj_pvd = ot.Project(input_file=prj_mechanics)
        # prj_pvd.save(tmp_path / "mechanics")
        # prj_test = ot.Project.from_folder(tmp_path/"mechanics")
        prj_test = prj_pvd.copy(tmp_path / "mechanics")
        prj_xdmf = ot.Project(input_file=prj_mechanics).copy()
        prj_xdmf.replace_text("XDMF", xpath="./time_loop/output/type")

        meshes_rect12 = ot.Meshes.from_gmsh(rect(n_edge_cells=12))
        meshes_rect10 = ot.Meshes.from_gmsh(rect(n_edge_cells=10))
        if save_strategy == "id":
            prj_pvd.save(id="pvd")
            prj_xdmf.save(id="xdmf")
            meshes_rect12.save(id="rect12")
            meshes_rect10.save(id="rect10")
        elif save_strategy == "target":
            prj_pvd.save(target=tmp_path / "pvd")
            prj_xdmf.save(target=tmp_path / "xdmf")
            meshes_rect12.save(path=tmp_path / "rect12")
            meshes_rect10.save(path=tmp_path / "rect10")

        prj_pvd_unsaved = ot.Project(input_file=prj_mechanics)
        model_pvd_rect12 = ot.Model(prj_pvd_unsaved, meshes=meshes_rect12)
        model_pvd_rect10 = ot.Model(prj_pvd, meshes_rect10)
        model_xdmf_rect10 = ot.Model(prj_xdmf, meshes=meshes_rect10)
        model_xdmf_rect12 = ot.Model(prj_xdmf, meshes=meshes_rect12)
        ot.Model(prj_test, meshes_rect12).save(tmp_path / "model_test1")
        ot.Model(prj_test, meshes_rect12).save(tmp_path / "model_test2")

        sim_default = model_pvd_rect10.run(id="sim_default")
        sim_highres = model_pvd_rect12.run(id="sim_highres")

        sim_xdmf = model_xdmf_rect10.run(id="sim_xdmf")
        assert sim_xdmf.result
        sim_xdmf_2 = model_xdmf_rect10.run()
        sim_xdmf_highres = model_xdmf_rect12.run()

        assert not model_pvd_rect10.user_specified_target
        assert not sim_default.is_saved

        sim_default.save()
        sim_highres.save()

        # 2 Sims just to compare but no long-term interest

        # different meshes make subtle differences
        ms_diff_rect = ot.MeshSeries.difference(
            sim_highres.result, sim_default.result
        )
        m_diff_12_displacement = abs(
            ms_diff_rect[-1].point_data["displacement"]
        )
        assert np.all(m_diff_12_displacement < 1e-3)
        assert not np.all(m_diff_12_displacement == 0)

        # execution of same model should give identical result
        ms_diff_run = ot.MeshSeries.difference(
            sim_xdmf.result, sim_xdmf_2.result
        )
        assert np.all(ms_diff_run[-1].point_data["displacement"] == 0)

        # same result with just other output format
        ms_diff_12_outputformat = ot.MeshSeries.difference(
            sim_default.result, sim_xdmf.result
        )
        assert np.all(
            ms_diff_12_outputformat[-1].point_data["displacement"] == 0
        )

        ms_diff_10_outputformat = ot.MeshSeries.difference(
            sim_highres.result, sim_xdmf_highres.result
        )
        assert np.all(
            ms_diff_10_outputformat[-1].point_data["displacement"] == 0
        )

        files = [
            f for f in meshes_rect10.active_target.glob("**/*") if f.is_file()
        ]
        # 5 meshes (1 domain + 4 sub) + 1 meta
        assert len(files) == 6

    @pytest.mark.parametrize("dry_run", [False, True])
    def test_save_returns_written_files(self, tmp_path, dry_run):
        """Test that all _save_impl methods return the actual written file paths.

        Tests both dry_run=False (actual save) and dry_run=True (simulation).
        In both cases, the same file paths should be returned.
        """

        # Test Project - folder with default.prj inside
        prj = ot.Project(input_file=prj_mechanics)
        files = prj.save(tmp_path / "test", overwrite=True, dry_run=dry_run)
        assert files[0] == tmp_path / "test" / "default.prj"
        assert_files_saved(files, expected_count=1, dry_run=dry_run)

        # Test Execution - single YAML file
        execution = ot.Execution()
        files = execution.save(
            tmp_path / "execution.yaml", overwrite=True, dry_run=dry_run
        )
        assert files[0] == tmp_path / "execution.yaml"
        assert_files_saved(files, expected_count=1, dry_run=dry_run)

        # Test Meshes - multiple VTU files + meta.yaml
        meshes = ot.Meshes.from_gmsh(rect(n_edge_cells=5))
        files = meshes.save(
            tmp_path / "test_meshes", overwrite=True, dry_run=dry_run
        )
        meta_file = tmp_path / "test_meshes" / "meta.yaml"
        assert meta_file in files, "meta.yaml should be in returned files"
        assert_files_saved(files, dry_run=dry_run)

        # Test Model - aggregates files from children
        model = ot.Model(prj, meshes)
        files = model.save(
            tmp_path / "test_model", overwrite=True, dry_run=dry_run
        )
        assert_files_saved(files, dry_run=dry_run)

        # Test MeshSeries - PVD file + VTU files
        ms = load_meshseries_HT_2D_VTU()
        files = ms.save(
            tmp_path / "test_series.pvd", overwrite=True, dry_run=dry_run
        )
        assert (
            files[0] == tmp_path / "test_series.pvd"
        ), "First file should be PVD"
        assert files[0].suffix == ".pvd"
        assert_files_saved(files, dry_run=dry_run)
