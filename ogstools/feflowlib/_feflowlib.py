# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

import logging as log

import ifm_contrib as ifm
import numpy as np
import pyvista as pv

ifm.forceLicense("Viewer")

# log configuration
logger = log.getLogger(__name__)


def points_and_cells(doc: ifm.FeflowDoc) -> tuple[np.ndarray, list, list]:
    """
    Get points and cells in a pyvista compatible format.

    :param doc: The FEFLOW data.
    :return: pts, cells, celltypes (points, cells, celltypes)
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
    dimension = doc.getNumberOfDimensions()
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
        celltypes.append(cell_type_dict[dimension][nElement])

    # 3. bring the elements to the right format for pyvista
    cells = np.array(elements).ravel()
    # 4 .write the list for all points and their global coordinates
    if dimension == 2:
        points = doc.c.mesh.df.nodes(global_cos=True)
        pts = points[["X", "Y"]].to_numpy()
        # A 0 is appended since in pyvista points must be given in 3D.
        # So we set the Z-coordinate to 0.
        pts = np.pad(pts, [(0, 0), (0, 1)])
        # order of points in the cells needs to be flipped
        if nElement == 3:
            cells = cells.reshape(-1, 4)
            cells[:, -3:] = np.flip(cells[:, -3:], axis=1)
            np.concatenate(cells)
        if nElement == 4:
            cells = cells.reshape(-1, 5)
            cells[:, -4:] = np.flip(cells[:, -4:], axis=1)
            np.concatenate(cells)
    elif dimension == 3:
        points = doc.c.mesh.df.nodes(
            global_cos=True, par={"Z": ifm.Enum.P_ELEV}
        )
        pts = points[["X", "Y", "Z"]].to_numpy()
    else:
        msg = "The input data is neither 2D nor 3D, which it needs to be."
        raise ValueError(msg)

    # 5. log information
    logger.info(
        "There are %s number of points and %s number of cells to be converted.",
        len(pts),
        len(celltypes),
    )
    return pts, cells, celltypes


def _material_ids_from_selections(
    doc: ifm.FeflowDoc,
) -> dict:
    """
    Get MaterialIDs from the FEFLOW data. Only applicable if they are
    saved in doc.c.sel.selections().

    :param doc: The FEFLOW data.
    :return: MaterialIDs
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
    logger.info("MaterialIDs refer to: %s", dict_matid)
    # MaterialIDs must be int32
    return {"MaterialIDs": np.array(mat_ids_mesh).astype(np.int32)}


def fetch_user_data(
    user_data: np.ndarray, geom_type: str, val_type: str
) -> list:
    return [
        data[1]
        for data in user_data
        if data[3] == geom_type and data[2] == val_type
    ]


def _point_and_cell_data(
    MaterialIDs: dict, doc: ifm.FeflowDoc
) -> tuple[dict, dict]:
    """
    Get point and cell data from Feflow data. Also write the MaterialIDs to the
    cell data.

    :param doc: The FEFLOW data.
    :param MaterialIDs:
    :return: pt_data, cell_data (point and cell data)
    """

    # 1. create a dictionary to filter all nodal and elemental values
    #  according to their name and ifm.Enum value
    items_pts = doc.c.mesh.df.get_available_items(Type="nodal")[
        "Name"
    ].to_dict()
    items_cell = doc.c.mesh.df.get_available_items(Type="elemental")[
        "Name"
    ].to_dict()
    # Get user data, which can be assigned manually in the FEFLOW file.
    # The exception is only needed, if there are no user data assigned.
    try:
        user_data = ifm.contrib_lib.user.UserPd(doc).distributions().to_numpy()
    except KeyError:
        user_data = []
    # 2. swap key and values in these dictionaries to have
    #  the name of the ifm.Enum value as key
    pts_dict = {y: x for x, y in items_pts.items()}
    cell_dict = {y: x for x, y in items_cell.items()}

    # 3. get all the nodal and elemental properties of the mesh as pandas
    # Dataframe and drop nans if a column is full of nans for all the properties
    pt_prop = doc.c.mesh.df.nodes(
        global_cos=True,
        par=pts_dict,
        distr=fetch_user_data(user_data, "NODAL", "DISTRIBUTION"),
        expr=fetch_user_data(user_data, "NODAL", "EXPRESSION"),
    ).dropna(axis=1, how="all")
    cell_prop = doc.c.mesh.df.elements(
        global_cos=True,
        par=cell_dict,
        distr=fetch_user_data(user_data, "ELEMENTAL", "DISTRIBUTION"),
        expr=fetch_user_data(user_data, "ELEMENTAL", "EXPRESSION"),
    ).dropna(axis=1, how="all")

    # 4. write the pandas Dataframe of nodal and elemental properties to
    #  a dictionary
    pt_data = pt_prop.to_dict("series")
    cell_data = cell_prop.to_dict("series")

    # 5. change format of cell data to a dictionary of lists
    cell_data = {key: [cell_data[key]] for key in cell_data}

    # 6. add MaterialIDs to cell data
    cell_data[str(list(MaterialIDs.keys())[0])] = [
        list(MaterialIDs.values())[0]
    ]

    # if P_LOOKUP_REGION is given and there are more different MaterialIDs given
    # than defined in selections, use P_LOOKUP_REGION for MaterialIDs
    if "P_LOOKUP_REGION" in cell_data and len(
        np.unique(MaterialIDs.values())
    ) < len(np.unique(cell_data["P_LOOKUP_REGION"])):
        cell_data["MaterialIDs"] = np.array(
            cell_data.pop("P_LOOKUP_REGION")
        ).astype(np.int32)

    # 7. write a list of all properties that have been dropped due to nans
    nan_arrays = [
        x
        for x in list(pts_dict.keys()) or list(cell_dict.keys())
        if x not in list(pt_data.keys()) and list(cell_data.keys())
    ]

    # 8. log the data arrays
    logger.info(
        "These data arrays refer to point data: %s", list(pt_data.keys())
    )
    logger.info(
        "These data arrays refer to cell data: %s", list(cell_data.keys())
    )
    logger.info(
        "These data arrays have been neglected as they are full of nans: %s",
        nan_arrays,
    )

    return (pt_data, cell_data)


def _convert_to_SI_units(mesh: pv.UnstructuredGrid) -> None:
    """
    FEFLOW often uses days as unit for time. In OGS SI-units are used.
    This is why days must be converted to seconds for properties to work
    correctly in OGS.

    :param mesh: mesh
    """

    arrays_to_be_converted = ["TRAF", "IOFLOW", "P_COND", "P_DIFF"]
    for data in list(mesh.point_data) + list(mesh.cell_data):
        if any(
            to_be_converted in data
            for to_be_converted in arrays_to_be_converted
        ):
            mesh[data] *= 1 / 86400
        elif "4TH" in data or "2ND" in data:
            mesh[data] *= -1 / 86400
        elif "HEAT" in data or "TEMP" in data:
            mesh[data] = mesh[data] + [273.15] * len(mesh[data])
    return mesh


def get_species_parameter(
    doc: ifm.FeflowDoc, mesh: pv.UnstructuredGrid
) -> tuple[dict, dict]:
    """
    Retrieve species parameters from FEFLOW data for points and cells.

    :param doc: The FEFLOW data.
    :param mesh: mesh
    :return: Dictionaries of point and cell species-specific data.
    """

    # Define common species parameters in FEFLOW.
    species_parameters = [
        "P_BC_MASS",
        "P_BCMASS_2ND",
        "P_BCMASS_3RD",
        "P_BCMASS_4TH",
        "P_CONC",
        "P_DECA",
        "P_DIFF",
        "P_LDIS",
        "P_PORO",
        "P_SORP",
        "P_TDIS",
        "P_TRAT_IN",
        "P_TRAT_OUT",
    ]

    species_point_dict: dict = {}
    species_cell_dict: dict = {}
    obsolete_data = {}

    data_dict = {"point": mesh.point_data, "cell": mesh.cell_data}
    species_dict = {"point": species_point_dict, "cell": species_cell_dict}
    for point_or_cell in ["point", "cell"]:
        for data in data_dict[point_or_cell]:
            if data in species_parameters:
                obsolete_data[data] = point_or_cell
                # If there is only a single species in the model, doc.getSpeciesName(i) throws a
                # RunTimeError.
                number_of_species = doc.getNumberOfSpecies()
                for species_id in range(number_of_species):
                    if number_of_species > 1:
                        species = doc.getSpeciesName(species_id)
                    else:
                        species = "single_species"
                    par = (
                        doc.getParameter(getattr(ifm.Enum, data), species)
                        if species != "single_species"
                        else doc.getParameter(getattr(ifm.Enum, data))
                    )
                    species_dict[point_or_cell][
                        species + "_" + data
                    ] = np.array(doc.getParamValues(par))

    return species_dict, obsolete_data


def _caclulate_retardation_factor(mesh: pv.UnstructuredGrid) -> None:
    """
    Calculates the retardation factor from the absorption coefficient, which is called
    Henry constant in FEFLOW, according to the formula: R = 1 + (1-p)/p * S. With R
    the retardation factor, p the porosity, S the absorption coefficient. Further details
    can be found in the FEFLOW book by Diersch in chapter 5.4.1.4 equation 5.70.

    :param mesh: mesh
    """
    for spec_porosity in [
        species_porosity
        for species_porosity in mesh.cell_data
        if "PORO" in species_porosity
    ]:
        porosity = mesh.cell_data[spec_porosity]
        species = spec_porosity.replace("P_PORO", "")
        # calculation of the retardation factor
        mesh.cell_data[species + "retardation_factor"] = (
            1 + mesh.cell_data[species + "P_SORP"] * (1 - porosity) / porosity
        )


def convert_geometry_mesh(doc: ifm.FeflowDoc) -> pv.UnstructuredGrid:
    """
    Get the geometric construction of the mesh.

    :param doc: The FEFLOW data.
    :return: mesh
    """
    points, cells, celltypes = points_and_cells(doc)
    return pv.UnstructuredGrid(cells, celltypes, points)


def update_geometry(
    mesh: pv.UnstructuredGrid, doc: ifm.FeflowDoc
) -> pv.UnstructuredGrid:
    """
    Update the geometric construction of the mesh with point and cell data.

    :param mesh: The mesh to be updated.
    :param doc: The FEFLOW data.
    :return: mesh
    """
    MaterialIDs = _material_ids_from_selections(doc)
    (point_data, cell_data) = _point_and_cell_data(MaterialIDs, doc)
    for pt_data in point_data:
        mesh.point_data.update({pt_data: point_data[pt_data]})
    for c_data in cell_data:
        mesh.cell_data.update({c_data: cell_data[c_data][0]})
    # If the FEFLOW problem class refers to a mass problem,
    # the following if statement will be true.
    if doc.getProblemClass() in [1, 3]:
        (
            species_dict,
            obsolete_data,
        ) = get_species_parameter(doc, mesh)
        for point_data in species_dict["point"]:
            mesh.point_data.update(
                {point_data: species_dict["point"][point_data]}
            )
        for cell_data in species_dict["cell"]:
            mesh.cell_data.update(
                {cell_data: species_dict["cell"][cell_data][0]}
            )
        for data, geometry in obsolete_data.items():
            if geometry == "point":
                mesh.point_data.remove(data)
            elif geometry == "cell":
                mesh.cell_data.remove(data)
            else:
                logger.error(
                    "Unknown geometry to remove obsolet data after conversion of chemical species."
                )
        _caclulate_retardation_factor(mesh)
    return _convert_to_SI_units(mesh)


def convert_properties_mesh(doc: ifm.FeflowDoc) -> pv.UnstructuredGrid:
    """
    Get the mesh with point and cell properties.

    :param doc: The FEFLOW data.
    :return: mesh
    """
    mesh = convert_geometry_mesh(doc)
    update_geometry(mesh, doc)
    return mesh
