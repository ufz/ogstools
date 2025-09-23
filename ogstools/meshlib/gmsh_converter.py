# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

import logging
from pathlib import Path

import meshio
import numpy as np
import pyvista as pv
from meshio._vtk_common import meshio_to_vtk_type

from .mesh import Mesh
from .subdomains import identify_subdomains

logging.basicConfig()  # Important, initializes root logger
logger = logging.getLogger(__name__)


def meshes_from_gmsh(
    filename: Path,
    dim: int | list[int] = 0,
    reindex: bool = True,
    log: bool = True,
    meshname: str = "domain",
) -> dict[str, Mesh]:
    """
    .. deprecated:: 0.7.1

    Use :class:`~ogstools.meshlib.meshes.Meshes`. :meth:`~ogstools.meshlib.meshes.Meshes.from_gmsh` instead.
    Generates pyvista unstructured grids from a gmsh mesh (.msh).

    Extracts domain-, boundary- and physical group-submeshes.

    :param filename:    Gmsh mesh file (.msh) as input data
    :param dim: Spatial dimension (1, 2 or 3), trying automatic detection,
                if not given. If multiple dimensions are provided, all elements
                of these dimensions are embedded in the resulting domain mesh.
    :param reindex: Physical groups / regions / Material IDs to be
                    renumbered consecutively beginning with zero.
    :param log:     If False, silence log messages
    :param meshame: The name of the domain mesh.

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

    # Workaround for pyvista 0.45
    # pv.from_meshio operates on mesh_cells, which
    # from 0.45 from_meshio expects (wrongly?) that physical groups are disjunct data sets
    # Workaround: cell_sets is temporary removed from data for from_meshio
    read_cells = mesh.cell_sets.copy()
    mesh.cell_sets = None
    pv_mesh: pv.UnstructuredGrid = pv.from_meshio(mesh).clean()
    mesh.cell_sets = read_cells

    # Code without workaround:
    # pv_mesh = pv.read(str(filename))
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

    meshes[meshname] = domain_mesh
    logger.info("%s: %s", "domain", domain_mesh)

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
            group_cells = np.full(pv_mesh.number_of_cells, False)
            for type_name, type_subset in mesh.cell_sets_dict[name].items():
                celltype = meshio_to_vtk_type[type_name]
                type_index = np.nonzero(pv_mesh.celltypes == celltype)[0]
                group_cells[type_index[type_subset]] = True
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
            msg = f"Unexpectedly got an empty mesh in physical group: {name}."
            raise RuntimeError(msg)

        sub_cell_dims = np.array([cell.dimension for cell in subdomain.cell])
        if not np.all(sub_cell_dims == sub_cell_dims[0]):
            msg = (
                f"Subdomain should only contain cells of dim {group_dim}, but"
                f"{name} contains cells of dim {np.unique(sub_cell_dims)}."
            )
            raise AssertionError(msg)

        subdomain.point_data.pop("vtkOriginalPointIds", None)
        subdomain.cell_data.pop("vtkOriginalCellIds", None)
        subdomain.point_data.pop("gmsh:dim_tags", None)
        subdomain.cell_data.pop("gmsh:physical", None)
        subdomain.cell_data.pop("gmsh:geometrical", None)
        subdomain.field_data.clear()
        identify_subdomains(domain_mesh, [subdomain])

        meshes[f"{name}"] = Mesh(subdomain)
        logger.info("%s: %s", f"{name}", subdomain)

    logger.info("Conversion complete.")

    return meshes
