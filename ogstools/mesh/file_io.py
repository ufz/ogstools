# SPDX-FileCopyrightText: Copyright (c) OpenGeoSys Community (opengeosys.org)
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv

from .ip_data import IPdata


def read(filename: Path | str) -> pv.UnstructuredGrid:
    "Read a single mesh from a filepath."
    return pv.read(filename)


from ogstools.core.storage import _date_temp_path


def save(
    mesh: pv.DataSet,
    filename: Path | str | None = None,
    drop_nan_keys: bool = False,
    **kwargs: Any,
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
        if not hasattr(pv, "set_new_attribute") or hasattr(mesh, "filepath"):
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
            if not hasattr(pv, "set_new_attribute") or hasattr(
                mesh, "filepath"
            ):
                mesh.filepath = outname
            else:
                pv.set_new_attribute(mesh, "filepath", outname)

    for data in [mesh.point_data, mesh.cell_data]:
        nan_keys = [k for k, v in data.items() if np.all(np.isnan(v))]
        for key in nan_keys:
            data.remove(key)

    ip_data = IPdata(mesh, auto_sync=False)
    if drop_nan_keys:
        nan_keys = [
            k for k, v in mesh.field_data.items() if np.all(np.isnan(v))
        ]
        for key in nan_keys:
            if key in ip_data._array_map:
                del ip_data[key]
            else:
                mesh.field_data.remove(key)
    ip_data._sync()

    if (
        outname.suffix == ".vtu"
        and not isinstance(mesh, pv.UnstructuredGrid)
        and hasattr(mesh, "cast_to_unstructured_grid")
    ):
        mesh.cast_to_unstructured_grid().save(outname, **kwargs)
    else:
        mesh.save(outname, **kwargs)

    return outname
