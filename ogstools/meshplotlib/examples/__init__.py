import importlib.resources as pkg_resources

from ogstools.meshlib import MeshSeries

# Will probably be replaced with some dynamically generated example

THM_2D_file = pkg_resources.files(__name__) / "2D.pvd"
meshseries_THM_2D = MeshSeries(str(THM_2D_file))
meshseries_CT_2D = MeshSeries(str(pkg_resources.files(__name__) / "elder.xdmf"))
meshseries_XDMF = MeshSeries(
    str(
        pkg_resources.files(__name__)
        / "2D_single_fracture_HT_2D_single_fracture.xdmf"
    )
)
