# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

from pathlib import Path
from typing import Any

import pyvista as pv


def read(filename: Path | str) -> pv.UnstructuredGrid:
    "Read a single mesh from a filepath."
    return pv.XMLUnstructuredGridReader(filename).read()


def save(
    filename: Path | str, mesh: pv.UnstructuredGrid, **kwargs: Any
) -> None:
    """Save mesh to file using ``pyvista.save_meshio``.

    Parameters
    :param filename:    Filename to save the mesh to
    :param mesh:        pyvista mesh
    """
    pv.save_meshio(filename, mesh, **kwargs)
