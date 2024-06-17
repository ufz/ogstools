from typing import cast

import pyvista as pv

from ogstools.definitions import SPATIAL_UNITS_KEY
from ogstools.propertylib import Scalar

from .plot_setup import PlotSetup
from .plot_setup_defaults import setup_dict

setup = PlotSetup.from_dict(setup_dict)


def spatial_quantity(mesh: pv.UnstructuredGrid | pv.DataSet) -> Scalar:
    "Return a Scalar able to transform the spatial units of the mesh."

    units = cast(
        list[int], mesh.field_data.get(SPATIAL_UNITS_KEY, [ord("m"), ord("m")])
    )
    data_unit, output_unit = (chr(unit) for unit in units)
    return Scalar("", data_unit, output_unit, "")
