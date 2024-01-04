from ogstools.definitions import ROOT_DIR
from ogstools.meshlib import MeshSeries

# Will probably be replaced with some dynamically generated example
examples = ROOT_DIR / "_examples"
meshseries_THM_2D = MeshSeries(str(examples / "2D.pvd"))
meshseries_CT_2D = MeshSeries(str(examples / "elder.xdmf"))
meshseries_XDMF = MeshSeries(
    str(examples / "2D_single_fracture_HT_2D_single_fracture.xdmf")
)
