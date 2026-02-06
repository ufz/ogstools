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
