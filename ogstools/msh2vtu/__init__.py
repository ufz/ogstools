# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

# Author: Dominik Kern (TU Bergakademie Freiberg)
import logging
from pathlib import Path
from typing import Any

import meshio
import numpy as np

logger = logging.getLogger(__name__)

# For more info on __all__ see: https://stackoverflow.com/a/35710527/80480
__all__ = [
    "my_remove_orphaned_nodes",
    "print_info",
    "find_cells_at_nodes",
    "find_connected_domain_cells",
    "msh2vtu",
]


def my_remove_orphaned_nodes(my_mesh: meshio.Mesh) -> None:
    """Auxiliary function to remove points not belonging to any cell"""

    # find connected points and derive mapping from all points to them
    connected_point_index = np.array([])
    for cell_block in my_mesh.cells:
        cell_block_values = cell_block.data
        connected_point_index = np.concatenate(
            [connected_point_index, cell_block_values.ravel()]
        ).astype(int)

    unique_point_index = np.unique(connected_point_index)
    old2new = np.zeros(len(my_mesh.points))
    for new_index, old_index in enumerate(unique_point_index):
        old2new[old_index] = int(new_index)

    # update mesh to remaining points
    my_mesh.points = my_mesh.points[unique_point_index]  # update points

    output_point_data = {}
    for pd_key, pd_value in my_mesh.point_data.items():
        output_point_data[pd_key] = pd_value[unique_point_index]
    my_mesh.point_data = output_point_data  # update point data

    output_cell_blocks = []
    for cell_block in my_mesh.cells:
        cell_type = cell_block.type
        cell_block_values = cell_block.data
        updated_values = old2new[cell_block_values].astype(int)
        output_cell_blocks.append(meshio.CellBlock(cell_type, updated_values))
    # cell data are not affected by point changes
    my_mesh.cells = output_cell_blocks  # update cells


# print info for mesh: statistics and data field names
def print_info(mesh: meshio.Mesh) -> None:
    N, D = mesh.points.shape
    logging.info("%d points in %d dimensions", N, D)
    cell_info = "cells: "
    for cell_type, cell_values in mesh.cells_dict.items():
        cell_info += str(len(cell_values)) + " " + cell_type + ", "
    logging.info(cell_info[0:-2])
    logging.info("point_data=%s", str(list(mesh.point_data)))
    logging.info("cell_data=%s", str(list(mesh.cell_data)))
    logging.info("cell_sets=%s", str(list(mesh.cell_sets)))
    logging.info("##")


# function to create node connectivity list, i.e. store for each node (point) to
# which element (cell) it belongs
def find_cells_at_nodes(
    cells: Any, node_count: int, cell_start_index: int
) -> list[set]:
    # depending on the numbering of mixed meshes in OGS one may think of an
    # object-oriented way to add elements (of different type) to node
    # connectivity

    # initialize list of sets
    node_connectivity: list[set] = [set() for _ in range(node_count)]
    cell_index = cell_start_index
    for cell in cells:
        for node in cell:
            node_connectivity[node].add(cell_index)
        cell_index += 1
    if node_connectivity.count(set()) > 0:
        unconnected_nodes = [
            node
            for node in range(node_count)
            if node_connectivity[node] == set()
        ]
        logging.info("Points not connected with domain cells:")
        logging.info(unconnected_nodes)
    return node_connectivity


def find_connected_domain_cells(
    boundary_cells_values: Any, domain_cells_at_node: list[set[int]]
) -> tuple[np.ndarray, np.ndarray]:
    "find out to which domain elements a boundary element belongs"
    # to return unique common connected domain cell to be stored as cell_data
    # ("bulk_element_id"), if there are more than one do not store anything
    domain_cells_array = np.zeros(len(boundary_cells_values))
    # number of connected domain_cells
    domain_cells_number = np.zeros(len(boundary_cells_values))

    # cell lists node of which it is comprised
    for cell_index, cell_values in enumerate(boundary_cells_values):
        connected_domain_cells = []
        for node in cell_values:
            connected_domain_cells.append(domain_cells_at_node[node])
        common_domain_cells = set.intersection(*connected_domain_cells)
        number_of_connected_domain_cells = len(common_domain_cells)
        domain_cells_number[cell_index] = number_of_connected_domain_cells
        # there should be one domain cell for each boundary cell, however cells
        # of boundary dimension may be in the domain (e.g. as sources)
        if number_of_connected_domain_cells == 1:
            # assign only one (unique) connected main cell
            domain_cells_array[cell_index] = common_domain_cells.pop()
    if (n_orphans := np.count_nonzero(domain_cells_number == 0)) > 0:
        logging.warning(
            "%s boundary_cells don't belong to any domain cell!", n_orphans
        )
    if (n_shared := np.count_nonzero(domain_cells_number > 1)) > 0:
        logging.info(
            "%s boundary_cells belong to more then one domain cell!", n_shared
        )

    return domain_cells_array, domain_cells_number


# TODO: move to Mesh read or similar
def msh2vtu(
    filename: Path,
    output_path: Path = Path(),
    output_prefix: str = "",
    dim: int | list[int] = 0,
    # TODO: below args can be moved to Mesh / write
    delz: bool = False,
    swapxy: bool = False,
    reindex: bool = False,
    keep_ids: bool = False,
    ascii: bool = False,
    log_level: int | str = "DEBUG",
) -> int:
    """
    Convert a gmsh mesh (.msh) to an unstructured grid file (.vtu).

    Prepares a Gmsh-mesh for use in OGS by extracting domain-,
    boundary- and physical group-submeshes, and saves them in
    vtu-format. Note that all mesh entities should belong to
    physical groups.

    :param filename:    Gmsh mesh file (.msh) as input data
    :param output_path: Path of output files, defaults to current working dir
    :param output_prefix: Output files prefix, defaults to basename of inputfile
    :param dim: Spatial dimension (1, 2 or 3), trying automatic detection,
                if not given. If multiple dimensions are provided, all elements
                of these dimensions are embedded in the resulting domain mesh.
    :param delz:    Delete z-coordinate, for 2D-meshes with z=0.
                    Note that vtu-format requires 3D points.
    :param swapxy:  Swap x and y coordinate
    :param reindex: Renumber physical group / region / Material IDs to be
                    renumbered beginning with zero.
    :param keep_ids:    By default, rename 'gmsh:physical' to 'MaterialIDs'
                        and change type of corresponding cell data to INT32.
                        If True, this is skipped.
    :param ascii:   Save output files (.vtu) in ascii format.
    :param log_level:   Level of log output. Possible values:
                        <https://docs.python.org/3/library/logging.html#levels>

    :returns: 0 if successful, otherwise error message.
    """
    logging.getLogger().setLevel(log_level)
    if isinstance(dim, list):
        assert len(dim) < 3, "Specify at most 3 dim values."
    filename = Path(filename)

    # ====== CONSTANTS ======
    # pylint: disable=invalid-name
    PH_INDEX = 0  # index of physical group id in field data of meshio objects
    GEO_INDEX = 1  # index of geometrical dim in field data of meshio objects
    DIM0 = 0
    DIM1 = 1
    DIM2 = 2
    DIM3 = 3
    # for all points, as the selection goes via the cells and subsequent trim
    OGS_POINT_DATA_KEY = "bulk_node_ids"
    # to associate domain points with original points
    OGS_DOMAIN_POINT_DATA_KEY = "original_node_number"
    AVAILABLE_CELL_TYPES = {
        DIM0: {"vertex"},
        DIM1: {"line", "line3", "line4"},
        DIM2: {
            *["triangle", "triangle6", "triangle9", "triangle10"],
            *["quad", "quad8", "quad9"],
        },
        DIM3: {
            *["tetra", "tetra10"],
            *["pyramid", "pyramid13", "pyramid15", "pyramid14"],
            *["wedge", "wedge15", "wedge18"],
            *["hexahedron", "hexahedron20", "hexahedron27"],
            *["prism", "prism15", "prism18"],
        },
    }
    # the following are cell data
    GMSH_PHYSICAL_KEY = "gmsh:physical"
    OGS_DOMAIN_KEY = "MaterialIDs"
    OGS_BOUNDARY_KEY = "bulk_elem_ids"
    # pylint: enable=invalid-name
    ogs = not keep_ids

    if not filename.is_file():
        logging.warning("No input file (mesh) found.")
        raise FileNotFoundError

    if filename.suffix != ".msh":
        logging.warning("Input file seems not to be in gmsh-format (*.msh)")

    # if no parameter given, use same basename as input file
    output_basename = filename.stem if output_prefix == "" else output_prefix
    logging.info("Output: %s", output_basename)

    mesh: meshio.Mesh = meshio.read(str(filename))
    points, point_data = mesh.points, mesh.point_data
    cells_dict, cell_data, cell_data_dict = (
        mesh.cells_dict,
        mesh.cell_data,
        mesh.cell_data_dict,
    )
    field_data = mesh.field_data
    number_of_original_points = len(points)
    existing_cell_types = set(mesh.cells_dict.keys())

    logging.info("Original mesh (read)")
    logging.info(mesh)

    # check if element types are supported in current version of this script
    all_available_cell_types: set[str] = set()
    for cell_types in AVAILABLE_CELL_TYPES.values():
        all_available_cell_types = all_available_cell_types.union(cell_types)
    for cell_type in existing_cell_types:
        if cell_type not in all_available_cell_types:
            logging.warning("Cell type %s not supported", str(cell_type))

    # set spatial dimension of mesh
    if dim == 0:
        assert isinstance(dim, int)
        # automatically detect spatial dimension of mesh
        _dim = DIM0
        for test_dim, test_cell_types in AVAILABLE_CELL_TYPES.items():
            if (
                len(test_cell_types.intersection(existing_cell_types)) > 0
                and test_dim > dim
            ):
                _dim = test_dim

        logging.info("Detected mesh dimension: %s", str(_dim))
        logging.info("##")
    else:
        # trust the user
        _dim = max(dim) if isinstance(dim, list) else dim

    # delete third dimension if wanted by user
    if delz:
        if _dim <= DIM2:
            logging.info("Remove z coordinate of all points.")
            points = mesh.points[:, :2]
        else:
            logging.info(
                "Mesh seems to be in 3D, z-coordinate cannot be removed. Option"
                " -z ignored."
            )

    # special case in 2D workflow
    if swapxy:
        logging.info("Swapping x- and y-coordinate")
        points[:, 0], points[:, 1] = points[:, 1], -points[:, 0]

    # boundary and domain cell types depend on dimension
    if DIM1 <= _dim <= DIM3:
        boundary_dim = _dim - 1
        domain_dim = _dim
        boundary_cell_types = existing_cell_types.intersection(
            AVAILABLE_CELL_TYPES[boundary_dim]
        )
        domain_cell_types = existing_cell_types.intersection(
            AVAILABLE_CELL_TYPES[domain_dim]
            if isinstance(dim, int)
            else set().union(*[AVAILABLE_CELL_TYPES[d] for d in dim])
        )
    else:
        logging.warning("Error, invalid dimension dim=%s!", str(_dim))
        return 1  # sys.exit()

    # Check for existence of physical groups
    if GMSH_PHYSICAL_KEY in cell_data:
        physical_groups_found = True

        # reconstruct field data, when empty (physical groups may have a number,
        # but no name)
        # TODO may there be other field data, than physical groups?
        if field_data == {}:
            # detect dimension by cell type
            for pg_cell_type, pg_cell_data in cell_data_dict[
                GMSH_PHYSICAL_KEY
            ].items():
                if pg_cell_type in AVAILABLE_CELL_TYPES[DIM0]:
                    pg_dim = DIM0
                if pg_cell_type in AVAILABLE_CELL_TYPES[DIM1]:
                    pg_dim = DIM1
                if pg_cell_type in AVAILABLE_CELL_TYPES[DIM2]:
                    pg_dim = DIM2
                if pg_cell_type in AVAILABLE_CELL_TYPES[DIM3]:
                    pg_dim = DIM3
                pg_ids = np.unique(pg_cell_data)
                for pg_id in pg_ids:
                    pg_key = "PhysicalGroup_" + str(pg_id)
                    field_data[pg_key] = np.array([pg_id, pg_dim])

        id_list_domains = np.unique(
            [
                ct_id
                for domain_cell_type in domain_cell_types
                for ct_id in cell_data_dict[GMSH_PHYSICAL_KEY][domain_cell_type]
            ]
        )
        id_map = {
            ph_id: re_id if reindex else ph_id
            for ph_id, re_id in zip(
                *np.unique(id_list_domains, return_inverse=True), strict=True
            )
        }

    else:
        logging.info("No physical groups found.")
        physical_groups_found = False

    ############################################################################
    # Extract domain mesh, note that meshio 4.3.3. offers
    # remove_lower_dimensional_cells(), but we want to keep a uniform style for
    # domain and subdomains. Make sure to use domain_mesh=deepcopy(mesh) in this
    # case!
    ############################################################################
    all_points = np.copy(points)  # copy all, superfluous get deleted later
    if ogs:
        # to associate domain points later
        original_point_numbers = np.arange(number_of_original_points)
        all_point_data = {}
        all_point_data[OGS_DOMAIN_POINT_DATA_KEY] = np.uint64(
            original_point_numbers
        )
    else:
        # deep copy
        all_point_data = {key: value[:] for key, value in point_data.items()}

    domain_cells = []
    domain_cell_data_key = OGS_DOMAIN_KEY if ogs else GMSH_PHYSICAL_KEY
    domain_cell_data: dict[str, list[int]] = {}
    domain_cell_data[domain_cell_data_key] = []

    for domain_cell_type in domain_cell_types:
        # cells
        domain_cells_values = cells_dict[domain_cell_type]
        number_of_domain_cells = len(domain_cells_values)
        domain_cells_block = (domain_cell_type, domain_cells_values)
        domain_cells.append(domain_cells_block)

        # cell_data
        domain_in_physical_group = physical_groups_found and (
            domain_cell_type in cell_data_dict[GMSH_PHYSICAL_KEY]
        )

        if domain_in_physical_group:
            domain_cell_data_values = cell_data_dict[GMSH_PHYSICAL_KEY][
                domain_cell_type
            ]
            if ogs:
                # ogs needs MaterialIDs as int32, possibly beginning with zero
                # (by id_offset)
                domain_cell_data_values = np.int32(
                    list(map(id_map.get, domain_cell_data_values))
                )

        else:
            domain_cell_data_values = np.zeros(
                (number_of_domain_cells), dtype=int
            )
            logging.info(
                "Some domain cells are not in a physical group, their"
                " PhysicalID/MaterialID is set to zero."
            )
        domain_cell_data[domain_cell_data_key].append(domain_cell_data_values)

    if len(domain_cells):
        domain_mesh = meshio.Mesh(
            points=all_points,
            point_data=all_point_data,
            cells=domain_cells,
            cell_data=domain_cell_data,
        )
        my_remove_orphaned_nodes(domain_mesh)

        if len(domain_mesh.points) != number_of_original_points:
            logging.warning(
                "There are nodes out of the domain mesh. If ogs option is set,"
                " then no bulk_node_id can be assigned to these nodes."
            )
        meshio.write(
            Path(output_path, output_basename + "_domain.vtu"),
            domain_mesh,
            binary=not ascii,
        )
        logging.info("Domain mesh (written)")
        print_info(domain_mesh)

        if ogs:
            # store domain node numbers for use as bulk_node_id (point_data)
            original2domain_point_table = (
                np.ones(number_of_original_points) * number_of_original_points
            )
            # initialize with non-existing number --> error when bulk_id for
            # non-domain mesh should be written
            for domain_point_index, original_point_index in enumerate(
                domain_mesh.point_data[OGS_DOMAIN_POINT_DATA_KEY]
            ):
                original2domain_point_table[
                    original_point_index
                ] = domain_point_index

        # prepare data needed for bulk_elem_id (cell_data), also needed without
        # ogs option to detect boundaries
        # node connectivity for a mixed mesh (check for OGS compliance), needed
        # with and without ogs option to identify boundary cells
        cell_start_index = 0
        domain_cells_at_node: list[set[int]] = [
            set() for _ in range(number_of_original_points)
        ]  # initialize list of sets
        # make a list for each type of domain cells
        for cell_block in domain_cells:
            block_domain_cells_at_node = find_cells_at_nodes(
                cell_block[1], number_of_original_points, cell_start_index
            )  # later used for boundary mesh and submeshes
            cell_start_index += len(
                cell_block[1]
            )  # assume consecutive cell numbering (as it is written to vtu)
            # add connectivities of current cell type to entries (sets) of total
            # connectivity (list of sets)
            for total_list, block_list in zip(
                domain_cells_at_node, block_domain_cells_at_node, strict=False
            ):
                total_list.update(block_list)

    else:
        logging.info("Empty domain mesh, nothing written to file.")

    ############################################################################
    # Extract boundary mesh
    ############################################################################

    # points, process full list (all points), later trimmed according to cell
    # selection, deep copy needed because removed_orphaned_nodes() operates on
    # shallow copy of point_data

    # copy again, in case previous remove_orphaned_nodes() affected all_points
    all_points = np.copy(points)

    if ogs:
        all_point_data = {}
        # now containing domain node numbers
        all_point_data[OGS_POINT_DATA_KEY] = np.uint64(
            original2domain_point_table
        )
    else:
        # deep copy
        all_point_data = {key: value[:] for key, value in point_data.items()}

    boundary_cells = []
    boundary_cell_data: dict[str, list] = {}
    boundary_cell_data_key = OGS_BOUNDARY_KEY if ogs else GMSH_PHYSICAL_KEY
    boundary_cell_data[boundary_cell_data_key] = []

    for boundary_cell_type in boundary_cell_types:
        # preliminary, as there may be cells of boundary dimension inside domain
        # (i.e. which are no boundary cells)
        boundary_cells_values = cells_dict[boundary_cell_type]
        connected_cells, connected_cells_count = (
            np.asarray(np.uint64(t))
            for t in find_connected_domain_cells(
                boundary_cells_values, domain_cells_at_node
            )
        )
        # a boundary cell is connected with exactly one domain cell
        boundary_index = connected_cells_count == 1
        # final boundary cells
        boundary_cells_values = boundary_cells_values[boundary_index]
        boundary_cells_block = (boundary_cell_type, boundary_cells_values)
        boundary_cells.append(boundary_cells_block)

        # cell_data
        boundary_in_physical_group = physical_groups_found and (
            boundary_cell_type in cell_data_dict[GMSH_PHYSICAL_KEY]
        )

        if ogs:
            boundary_cell_data_values = connected_cells[boundary_index]
        else:
            if boundary_in_physical_group:
                boundary_cell_data_values = cell_data_dict[GMSH_PHYSICAL_KEY][
                    boundary_cell_type
                ]
            else:
                # cells of specific type
                number_of_boundary_cells = len(boundary_cells_values)
                boundary_cell_data_values = np.zeros(
                    (number_of_boundary_cells), dtype=int
                )
                logging.info(
                    "Some boundary cells are not in a physical group, their"
                    " PhysicalID is set to zero."
                )
        boundary_cell_data[boundary_cell_data_key].append(
            boundary_cell_data_values
        )

    boundary_mesh = meshio.Mesh(
        points=all_points,
        point_data=all_point_data,
        cells=boundary_cells,
        cell_data=boundary_cell_data,
    )
    if len(boundary_cells):
        my_remove_orphaned_nodes(boundary_mesh)

        meshio.write(
            Path(output_path, output_basename + "_boundary.vtu"),
            boundary_mesh,
            binary=not ascii,
        )
        logging.info("Boundary mesh (written)")
        print_info(boundary_mesh)
    else:
        logging.info("No boundary elements detected.")

    ############################################################################
    # Now we want to extract subdomains given by physical groups in gmsh
    # name=user-defined name of physical group, data=[physical_id, geometry_id]
    ############################################################################
    if not physical_groups_found:
        return 0

    for name, data in field_data.items():
        ph_id = data[PH_INDEX]  # only used for old versions of MSH
        subdomain_dim = data[GEO_INDEX]
        if subdomain_dim >= DIM0 and subdomain_dim <= DIM3:
            subdomain_cell_types = existing_cell_types.intersection(
                AVAILABLE_CELL_TYPES[subdomain_dim]
            )
        else:
            logging.warning("Invalid dimension found in physical groups.")
            continue

        all_points = np.copy(points)
        # point data, make another copy due to possible changes by previous
        # actions
        if ogs:
            all_point_data = {}
            all_point_data[OGS_POINT_DATA_KEY] = np.uint64(
                original2domain_point_table
            )
        else:
            all_point_data = {
                key: value[:] for key, value in point_data.items()
            }  # deep copy

        # cells, cell_data
        subdomain_cells = []  # list
        subdomain_cell_data: dict[str, list] = {}  # dict
        if ogs:
            if subdomain_dim == domain_dim:
                subdomain_cell_data_key = OGS_DOMAIN_KEY
            elif subdomain_dim == boundary_dim:
                subdomain_cell_data_key = OGS_BOUNDARY_KEY
            else:
                # use gmsh, as the requirements from OGS
                subdomain_cell_data_key = GMSH_PHYSICAL_KEY
        else:
            # same for all dimensions
            subdomain_cell_data_key = GMSH_PHYSICAL_KEY
        subdomain_cell_data[subdomain_cell_data_key] = []  # list
        # flag to indicate invalid bulk_element_ids, then no cell data will be
        # written
        subdomain_cell_data_trouble = False

        for cell_type in subdomain_cell_types:
            # cells
            all_false = np.full(
                cell_data_dict[GMSH_PHYSICAL_KEY][cell_type].shape,
                False,
            )
            if mesh.cell_sets_dict != {}:
                selection_index = mesh.cell_sets_dict[name].get(
                    cell_type, all_false
                )
            else:
                selection_index = (
                    cell_data_dict[GMSH_PHYSICAL_KEY][cell_type] == ph_id
                )
            selection_cells_values = cells_dict[cell_type][selection_index]
            if len(selection_cells_values):  # if there are some data
                selection_cells_block = (cell_type, selection_cells_values)
                subdomain_cells.append(selection_cells_block)

                # cell data
                if ogs:
                    selection_cell_data_values: np.int32 | np.uint64
                    if subdomain_dim == boundary_dim:
                        (
                            connected_cells,
                            connected_cells_count,
                        ) = find_connected_domain_cells(
                            selection_cells_values, domain_cells_at_node
                        )
                        # a boundary cell is connected with one domain cell,
                        # needed to write bulk_elem_id
                        boundary_index = connected_cells_count == 1
                        selection_cell_data_values = np.uint64(connected_cells)
                        if not boundary_index.all():
                            logging.info(
                                "In physical group %s"
                                " are bulk_elem_ids not uniquely defined,"
                                " e.g. for cells of boundary dimension inside"
                                " the domain, and thus not written. If"
                                " bulk_elem_ids should be written for a"
                                " physical group, then make sure all its"
                                " cells of boundary dimension are located at"
                                " the boundary.",
                                name,
                            )
                            subdomain_cell_data_trouble = True
                    elif subdomain_dim == domain_dim:
                        selection_cell_data_values = np.int32(
                            cell_data_dict[GMSH_PHYSICAL_KEY][cell_type][
                                selection_index
                            ]
                        )
                        selection_cell_data_values = list(
                            map(id_map.get, selection_cell_data_values)
                        )

                    else:  # any cells of lower dimension than boundary
                        selection_cell_data_values = np.int32(
                            cell_data_dict[GMSH_PHYSICAL_KEY][cell_type][
                                selection_index
                            ]
                        )

                else:
                    selection_cell_data_values = cell_data_dict[
                        GMSH_PHYSICAL_KEY
                    ][cell_type][selection_index]

                subdomain_cell_data[subdomain_cell_data_key].append(
                    selection_cell_data_values
                )

        outputfilename = output_basename + "_physical_group_" + name + ".vtu"
        if subdomain_cell_data_trouble:
            submesh = meshio.Mesh(
                points=all_points,
                point_data=all_point_data,
                cells=subdomain_cells,
            )  # do not write invalid cell_data
        else:
            submesh = meshio.Mesh(
                points=all_points,
                point_data=all_point_data,
                cells=subdomain_cells,
                cell_data=subdomain_cell_data,
            )

        if len(subdomain_cells) > 0:
            my_remove_orphaned_nodes(submesh)

            outputfilename = (
                output_basename + "_physical_group_" + name + ".vtu"
            )
            meshio.write(
                Path(output_path, outputfilename), submesh, binary=not ascii
            )
            logging.info("Submesh %s (written)", name)
            print_info(submesh)
        else:
            logging.info("Submesh %s empty (not written)", name)

    return 0  # successfully finished
