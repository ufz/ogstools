# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

# Original author: Dominik Kern (TU Bergakademie Freiberg)
# Author: Florian Zill

import logging
from pathlib import Path

import meshio
import numpy as np
import pyvista as pv

from .mesh import Mesh

logging.basicConfig()  # Important, initializes root logger
logger = logging.getLogger(__name__)


def meshes_from_gmsh(
    filename: Path,
    dim: int | list[int] = 0,
    reindex: bool = True,
    log: bool = True,
) -> dict[str, Mesh]:
    """
    Generates pyvista unstructured grids from a gmsh mesh (.msh).

    Extracts domain-, boundary- and physical group-submeshes.

    :param filename:    Gmsh mesh file (.msh) as input data
    :param dim: Spatial dimension (1, 2 or 3), trying automatic detection,
                if not given. If multiple dimensions are provided, all elements
                of these dimensions are embedded in the resulting domain mesh.
    :param reindex: Physical groups / regions / Material IDs to be
                    renumbered consecutively beginning with zero.
    :param log:     If False, silence log messages

    :returns: A dictionary of names and corresponding meshes
    """
    logger.setLevel(logging.INFO if log else logging.ERROR)

    if isinstance(dim, list) and len(dim) > 3:
        msg = "Specify at most 3 dim values."
        raise ValueError(msg)
    filename = Path(filename)

    if not filename.is_file():
        raise FileNotFoundError

    meshes: dict[str, meshio.Mesh] = {}
    mesh: meshio.Mesh = meshio.read(str(filename))
    pv_mesh = pv.from_meshio(mesh).clean()
    if "gmsh:physical" not in pv_mesh.cell_data:
        pv_mesh.cell_data["gmsh:physical"] = np.zeros(pv_mesh.number_of_cells)

    pv_cell_dims = np.asarray([cell.dimension for cell in pv_mesh.cell])
    if dim == 0:
        dim = [np.max(pv_cell_dims)]
        logger.info("Detected domain dimension of %d", dim[0])
    elif isinstance(dim, int):
        dim = [dim]
    domain_mesh = Mesh(
        pv_mesh.extract_cells([cell.dimension in dim for cell in pv_mesh.cell])
    )
    mat_ids = domain_mesh.cell_data.pop(
        "gmsh:physical", np.zeros(domain_mesh.number_of_cells)
    )
    unique_mat_ids = np.unique(mat_ids)
    logger.info("Found material IDs: %s", unique_mat_ids)
    domain_mesh.clear_cell_data()
    domain_mesh.clear_point_data()
    domain_mesh.clear_field_data()
    domain_mesh.cell_data["MaterialIDs"] = np.int32(mat_ids)
    if reindex:
        domain_mesh.reindex_material_ids()
        logger.info("Renumbered to: %s", np.unique(domain_mesh["MaterialIDs"]))

    meshes["domain"] = domain_mesh
    logger.info("%s: %s", "domain", domain_mesh)

    group_ids = np.unique(pv_mesh["gmsh:physical"])
    for name, (group_index, group_dim) in mesh.field_data.items():
        # skip iteration for domain mesh
        if name == filename.stem:
            continue

        # for old gmsh versions
        if mesh.cell_sets == {}:
            subdomain = pv_mesh.extract_cells(
                pv_cell_dims == group_dim
            ).threshold([group_index, group_index], "gmsh:physical")
        # for recent gmsh versions (allows cells belonging to multiple groups)
        else:
            # Gmsh may store element of the same physical id in different
            # blocks. To mark all those elements correctly for extraction in the
            # pyvista mesh, we need to add an offset of the running total of
            # counted cells, as we don't have these blocks in the pyvista mesh.
            group_offsets = {index: 0 for index in group_ids}
            group_cells = np.full(pv_mesh.number_of_cells, False)
            for index, cell_ids in enumerate(mesh.cell_sets[name]):
                if len(cell_ids) == 0:
                    continue
                # assert all in this array are the same
                group_id = mesh.cell_data["gmsh:physical"][index][0]
                select = np.argwhere(
                    (pv_mesh["gmsh:physical"] == group_id)
                    & (pv_cell_dims == group_dim)
                )[:, 0]
                group_cells[select[cell_ids + group_offsets[group_id]]] = True
                group_offsets[group_id] += len(
                    mesh.cell_data["gmsh:physical"][index]
                )
            subdomain = pv_mesh.extract_cells(group_cells)

        # Workaround for gmsh python api bug
        # (physical group with tag 0 doesn't produce correct msh file)
        if (
            subdomain.number_of_cells == 0
            and group_index == 1
            and 0 in pv_mesh["gmsh:physical"]
        ):
            subdomain = pv_mesh.threshold([0, 0], "gmsh:physical")
        if subdomain.number_of_cells == 0:
            msg = "Unexpectedly got an empty mesh."
            raise RuntimeError(msg)

        sub_cell_dims = np.array([cell.dimension for cell in subdomain.cell])
        if not np.all(sub_cell_dims == sub_cell_dims[0]):
            msg = (
                f"Subdomain should only contain cells of dim {group_dim}, but"
                f"{name} contains cells of dim {np.unique(sub_cell_dims)}."
            )
            raise AssertionError(msg)

        # rename vtk fields to OGS conventions and change to correct type
        subdomain["bulk_elem_ids"] = np.uint64(
            subdomain.cell_data.pop("vtkOriginalCellIds")
        )
        subdomain["bulk_node_ids"] = np.uint64(
            subdomain.point_data.pop("vtkOriginalPointIds")
        )

        # remove unnecessary data
        for cell_data_key in ["gmsh:physical", "gmsh:geometrical"]:
            if cell_data_key in subdomain.cell_data:
                subdomain.cell_data.remove(cell_data_key)
        if "gmsh:dim_tags" in subdomain.point_data:
            subdomain.point_data.remove("gmsh:dim_tags")
        subdomain.field_data.clear()

        meshes[f"physical_group_{name}"] = Mesh(subdomain)
        logger.info("%s: %s", f"physical_group_{name}", subdomain)

    logger.info("Conversion complete.")

    return meshes
