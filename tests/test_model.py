import ogstools as ot
from ogstools.examples import EXAMPLES_DIR


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
