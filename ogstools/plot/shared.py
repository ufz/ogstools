import pyvista as pv

from ogstools.definitions import SPATIAL_UNITS_KEY
from ogstools.propertylib import Scalar

from .plot_setup import PlotSetup
from .plot_setup_defaults import setup_dict

setup = PlotSetup.from_dict(setup_dict)


def spatial_quantity(mesh: pv.UnstructuredGrid) -> Scalar:
    "Return a Scalar able to transform the spatial units of the mesh."

    spatial_units = mesh.field_data.get(SPATIAL_UNITS_KEY, ["m", "m"])
    return Scalar("", spatial_units[0], spatial_units[1], "")
