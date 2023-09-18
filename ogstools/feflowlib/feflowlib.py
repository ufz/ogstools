"""
Created on Tue Mar 14 2023

@author: heinzej
"""

import logging as log
from sys import stdout

import ifm_contrib as ifm
import numpy as np
import pyvista as pv

# log configuration
log.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
    level=log.INFO,
    stream=stdout,
    datefmt="%d/%m/%Y %H:%M:%S",
)

ifm.forceLicense("Viewer")

# log feflow version
log.info(
    "The converter is working with FEFLOW %s (build %s).",
    ifm.getKernelVersion() / 1000,
    ifm.getKernelRevision(),
)


def points_and_cells(doc: ifm.FeflowDoc):
    """
    Get points and cells in a pyvista compatible format.

    :param doc: The FEFLOW data.
    :type doc: ifm.FeflowDoc
    :return: pts, cells, celltypes (points, cells, celltypes)
    :rtype: tuple(numpy.ndarray, list, list)
    """
    # 0. define variables
    cell_type_dict = {
        2: {3: pv.CellType.TRIANGLE, 4: pv.CellType.QUAD},
        3: {
            4: pv.CellType.TETRA,
            6: pv.CellType.WEDGE,
            8: pv.CellType.HEXAHEDRON,
        },
    }
    # 1. get a list of all cells/elements and reverse it for correct node order in OGS
    elements = np.fliplr(np.array(doc.c.mesh.get_imatrix())).tolist()
    # 2. write the amount of nodes per element to the first entry of each list
    # of nodes. This is needed for pyvista !
    # 2 could also be done with np.hstack([len(ele)]*len(elements,elements))
    # also write the celltypes.
    celltypes = []
    for element in elements:
        nElement = len(element)
        element.insert(0, nElement)
        celltypes.append(cell_type_dict[doc.getNumberOfDimensions()][nElement])

    # 3. bring the elements to the right format for pyvista
    cells = np.array(elements).ravel()
    # 4 .write the list for all points and their global coordinates
    points = doc.c.mesh.df.nodes(global_cos=True, par={"Z": ifm.Enum.P_ELEV})
    pts = points[["X", "Y", "Z"]].to_numpy()

    # 5. log information
    log.info(
        "There are %s number of points and %s number of cells to be converted.",
        len(pts),
        len(celltypes),
    )

    return pts, cells, celltypes


def material_ids_from_selections(doc: ifm.FeflowDoc):
    """
    Get MaterialIDs from the FEFLOW data. Only applicable if they are
    saved in doc.c.sel.selections().

    :param doc: The FEFLOW data.
    :type doc: ifm.FeflowDoc
    :return: MaterialIDs
    :rtype: tuple
    """
    # Note: an error occurs if there are no elements defined to the selection

    # 1. define necessary variables
    mat_ids = []
    elements = []
    dict_matid = {}
    mat_id = 0

    # 2. try to overcome ValueError for selec not referring to element
    for selec in doc.c.sel.selections():
        try:
            # 2.1 write a list of elements that refer to a material that
            # is defined in selections (mat__elements)
            # this call can cause an ValueError since not all selections refer to elements
            # but all elemental selection refer to material_ids
            elements.append(
                doc.c.mesh.df.elements(selection=selec).index.values
            )
            # 2.2 write a list with a corresponding id for
            # that material (mat_id)
            mat_ids.append(int(mat_id))
            # 2.3 write a dictionary to understand which id refers
            # to which selection
            dict_matid[mat_id] = selec
            mat_id += 1
        except ValueError:
            pass

    # 3. write a list of material ids. The id is written to the corresponding
    # entry in the list.
    mat_ids_mesh = [0] * doc.getNumberOfElements()

    for count, value in enumerate(mat_ids):
        for element in elements[count]:
            mat_ids_mesh[element] = value

    # 4. log the dictionary of the MaterialIDs
    log.info("MaterialIDs refer to: %s", dict_matid)
    # MaterialIDs must be int32
    return {"MaterialIDs": np.array(mat_ids_mesh).astype(np.int32)}


def point_and_cell_data(MaterialIDs: dict, doc: ifm.FeflowDoc):
    """
    Get point and cell data from Feflow data. Also write the MaterialIDs to the
    cell data.

    :param doc: The FEFLOW data.
    :type doc: ifm.FeflowDoc
    :param MaterialIDs:
    :type MaterialIDs: dict
    :return: pt_data, cell_data (point and cell data)
    :rtype: tuple(dict,dict)
    """

    # 1. create a dictionary to filter all nodal and elemental values
    #  according to their name and ifm.Enum value
    items_pts = doc.c.mesh.df.get_available_items(Type="nodal")[
        "Name"
    ].to_dict()
    items_cell = doc.c.mesh.df.get_available_items(Type="elemental")[
        "Name"
    ].to_dict()

    # 2. swap key and values in these dictionaries to have
    #  the name of the ifm.Enum value as key
    pts_dict = {y: x for x, y in items_pts.items()}
    cell_dict = {y: x for x, y in items_cell.items()}

    # 3. get all the nodal and elemental properties of the mesh as pandas
    # Dataframe and drop nans if a column is full of nans for all the properties
    pt_prop = doc.c.mesh.df.nodes(global_cos=True, par=pts_dict).dropna(
        axis=1, how="all"
    )
    cell_prop = doc.c.mesh.df.elements(global_cos=True, par=cell_dict).dropna(
        axis=1, how="all"
    )

    # 4. write the pandas Dataframe of nodal and elemental properties to
    #  a dictionary
    pt_data = pt_prop.to_dict("series")
    cell_data = cell_prop.to_dict("series")

    # 5. change format of cell data to a dictionary of lists
    cell_data = {key: [cell_data[key]] for key in cell_data}

    # 6. add materialIDs to cell data
    cell_data[str(list(MaterialIDs.keys())[0])] = [
        list(MaterialIDs.values())[0]
    ]

    # 7. write a list of all properties that have been dropped due to nans
    nan_arrays = [
        x
        for x in list(pts_dict.keys()) or list(cell_dict.keys())
        if x not in list(pt_data.keys()) and list(cell_data.keys())
    ]

    # 8. log the data arrays
    log.info("These data arrays refer to point data: %s", list(pt_data.keys()))
    log.info("These data arrays refer to cell data: %s", list(cell_data.keys()))
    log.info(
        "These data arrays have been neglected as they are full of nans: %s",
        nan_arrays,
    )

    return pt_data, cell_data


def read_geometry(doc: ifm.FeflowDoc):
    """
    Get the geometric construction of the mesh.

    :param doc: The FEFLOW data.
    :type doc: ifm.FeflowDoc
    :return: mesh
    :rtype: pyvista.UnstructuredGrid
    """
    points, cells, celltypes = points_and_cells(doc)
    return pv.UnstructuredGrid(cells, celltypes, points)


def update_geometry(mesh: pv.UnstructuredGrid, doc: ifm.FeflowDoc):
    """
    Update the geometric construction of the mesh with point and cell data.

    :param mesh: The mesh to be updated.
    :type mesh: pyvista.UnstructuredGrid
    :param doc: The FEFLOW data.
    :type doc: ifm.FeflowDoc
    :return: mesh
    :rtype: pyvista.UnstructuredGrid
    """
    MaterialIDs = material_ids_from_selections(doc)
    point_data, cell_data = point_and_cell_data(MaterialIDs, doc)
    for i in point_data:
        mesh.point_data.update({i: point_data[i]})
    for i in cell_data:
        mesh.cell_data.update({i: cell_data[i][0]})
    return mesh


def read_properties(doc: ifm.FeflowDoc):
    """
    Get the mesh with point and cell properties.

    :param doc: The FEFLOW data.
    :type doc: ifm.FeflowDoc
    :return: mesh
    :rtype: pyvista.UnstructuredGrid
    """
    mesh = read_geometry(doc)
    update_geometry(mesh, doc)
    return mesh
