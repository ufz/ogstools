import importlib.resources as pkg_resources

from ogstools.meshlib import MeshSeries

# Will probably be replaced with some dynamically generated example
meshseries_THM_2D = MeshSeries(str(pkg_resources.files(__name__) / "2D.pvd"))
meshseries_CT_2D = MeshSeries(str(pkg_resources.files(__name__) / "elder.xdmf"))
