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


from ogstools.core.storage import _date_temp_path


def save(
    mesh: pv.DataSet, filename: Path | str | None = None, **kwargs: Any
) -> Path:
    """Save mesh to file.

    Supported are all file formats pyvista supports.
    In case you want to save as a vtu-file and the given mesh is not a
    `pv.UnstructuredGrid` it is cast to one prior to saving.

    :param mesh:        pyvista mesh
    :param filename:    Filepath to save the mesh to

    :return:            Filepath to saved mesh
    """
    if filename:
        filename = Path(filename)
        if hasattr(mesh, "filepath"):
            mesh.filepath = filename
        else:
            pv.set_new_attribute(mesh, "filepath", filename)
        outname = Path(filename)
    else:
        existing = getattr(mesh, "filepath", None)
        if existing:
            outname = Path(existing)
        else:
            # invent a generic filename
            outname = _date_temp_path("Mesh", "vtu")
            outname.parent.mkdir(exist_ok=True, parents=True)
            if hasattr(mesh, "filepath"):
                mesh.filepath = outname
            else:
                pv.set_new_attribute(mesh, "filepath", outname)

    if (
        outname.suffix == ".vtu"
        and not isinstance(mesh, pv.UnstructuredGrid)
        and hasattr(mesh, "cast_to_unstructured_grid")
    ):
        mesh.cast_to_unstructured_grid().save(outname, **kwargs)
    else:
        mesh.save(outname, **kwargs)

    return outname
