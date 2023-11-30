import importlib.resources as pkg_resources

# Will probably be replaced with some dynamically generated example
pvd_file = str(pkg_resources.files(__name__) / "2D.pvd")
xdmf_file = str(
    pkg_resources.files(__name__)
    / "2D_single_fracture_HT_2D_single_fracture.xdmf"
)
