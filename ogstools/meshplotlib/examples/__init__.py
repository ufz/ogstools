import importlib.resources as pkg_resources

from ..mesh_series import MeshSeries

# Will probably be replaced with some dynamically generated example
meshseries_THM_2D = MeshSeries(str(pkg_resources.files(__name__) / "2D.pvd"))
