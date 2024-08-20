from typing import cast

import pyvista as pv

from ogstools.definitions import SPATIAL_UNITS_KEY
from ogstools.variables import Scalar

from .plot_setup import PlotSetup
from .plot_setup_defaults import setup_dict

setup = PlotSetup.from_dict(setup_dict)


def spatial_quantity(mesh: pv.UnstructuredGrid | pv.DataSet) -> Scalar:
    "Return a Scalar able to transform the spatial units of the mesh."

    units = cast(
        list[int],
        mesh.field_data.get(SPATIAL_UNITS_KEY, [ord(c) for c in "m,m"]),
    )
    data_unit, output_unit = "".join(chr(unit) for unit in units).split(",")
    return Scalar("", data_unit, output_unit, "")
