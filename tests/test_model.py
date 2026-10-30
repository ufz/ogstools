import shutil

import pytest
import pyvista as pv

import ogstools as ot
from ogstools import examples
from ogstools.definitions import EXAMPLES_DIR


def test_model_construct_without_explicit_meshes():
    """
    Intended to be default reading of models store in OGS Tests/Data
    Here the meshes and prj file are flat in the same folder
    """
    prj_file = (
        EXAMPLES_DIR
        / "prj"
        / "TH2M"
        / "H2M"
        / "Liakopoulos"
        / "liakopoulos_TH2M.prj"
    )
    m = ot.Model(prj_file)
    assert len(m.meshes) == 5


@pytest.mark.system  # ogs --version check on model.save()
def test_model_deep_copy_with_changed_project(tmp_path):
    prj_file = EXAMPLES_DIR / "prj" / "simple_mechanics.prj"
    model = ot.Model(prj_file)
    model.save(tmp_path / "test_model_deep_copy_with_changed_project")

    model3 = model.copy()
    assert model3.project is not model.project
    assert model3.project == model.project

    prj2 = model.project.copy()
    prj2.processes.add_process_variable("test", "test")
    modelc = ot.Model(prj2, model.meshes)
    assert model.project != modelc.project
    modelc.save(tmp_path / "test_model_deep_copy_with_changed_project_copy")
    prj_text = model.project.prjfile.read_text()
    prj2_text = modelc.project.prjfile.read_text()
    assert "<test>test</test>" in prj2_text
    assert "<test>test</test>" not in prj_text


def test_model_construct_without_explicit_meshes2():
    """
    Intended to be default reading of models stored with OGSTools
    Here the meshes and prj file are flat in the same folder
    """
    prj_file = (
        EXAMPLES_DIR
        / "simulation"
        / "small_deformation"
        / "model"
        / "default.prj"
    )
    m = ot.Model(prj_file)
    assert len(m.meshes) == 5


def test_domain_mesh_resolution(tmp_path):
    mesh = pv.Plane().cast_to_unstructured_grid()
    prj = ot.Project(output_file=tmp_path / "default.prj")

    # first one is domain mesh
    for name in ["square_quad8", "domain"]:
        path = tmp_path / f"{name}.vtu"
        mesh.save(path)
        prj.mesh.add_mesh(path)
    prj.write_input()

    model = ot.Model(tmp_path / "default.prj")
    assert model.meshes.domain_name == "square_quad8"


@pytest.mark.system
def test_model_run_python_bc_imports_module_next_to_it(tmp_path):
    """The python BC script imports decay_parameters.py, which lies next to it."""
    from ogstools.examples import prj_nuclear_decay

    meshes = ot.Meshes.from_gmsh(
        ot.gmsh_tools.rect(lengths=100.0, n_edge_cells=(5, 1)), log=False
    )
    sim = ot.Model(ot.Project(prj_nuclear_decay), meshes).run(tmp_path / "sim")
    assert sim.status == sim.Status.done, sim.status_str


@pytest.mark.tools  # constructMeshesFromGeometry
def test_model_constraint_meshes_apart_from_prj(tmp_path):
    """Geometry subdomains are built when the meshes do not lie next to the prj file.

    scope: component (Model._constraint_meshes) | oracle: correctness, subdomains
    """
    prj_dir = tmp_path / "prj"
    prj_dir.mkdir()
    for file in [
        examples.prj_two_anchors,
        examples.prj_two_anchors.with_name("square_1x1.gml"),
    ]:
        shutil.copy(file, prj_dir)
    model = ot.Model(
        prj_dir / examples.prj_two_anchors.name,
        meshes=examples.prj_two_anchors.parent,
    )
    meshes = model._constraint_meshes()
    # top, bottom: built from square_1x1.gml, as the prj uses them
    # two_anchors: read from the meshes folder
    assert set(meshes.subdomains) == {"top", "bottom", "two_anchors"}
