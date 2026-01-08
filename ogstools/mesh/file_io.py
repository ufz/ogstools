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
    return pv.read(filename)


def save(filename: Path | str, mesh: pv.DataSet, **kwargs: Any) -> None:
    """Save mesh to file.

    Supported are all file formats pyvista supports.
    In case you want to save as a vtu-file and the given mesh is not a
    `pv.UnstructuredGrid` it is cast to one prior to saving.

    :param filename:    Filename to save the mesh to
    :param mesh:        pyvista mesh
    """
    if (
        Path(filename).suffix == ".vtu"
        and not isinstance(mesh, pv.UnstructuredGrid)
        and hasattr(mesh, "cast_to_unstructured_grid")
    ):
        mesh.cast_to_unstructured_grid().save(filename, **kwargs)
    else:
        mesh.save(filename, **kwargs)
