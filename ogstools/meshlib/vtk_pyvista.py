# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

import numpy as np
import vtk


def cell_points(cell_type: int) -> int:
    """
    Return the number of points for a given VTK fixed-size cell type.

    Parameters
    ----------
    cell_type : int
        VTK cell type ID (e.g., vtk.VTK_LINE, vtk.VTK_TRIANGLE, etc.)

    Returns
    -------
    int
        Number of points for this cell type.

    Raises
    ------
    AttributeError
        If the cell type is variable-sized (e.g., POLY_LINE, POLYGON),
        since their number of points is not fixed.

    Example
    -------
    >>> import vtk
    >>> cell_points(vtk.VTK_LINE)
    2
    >>> cell_points(vtk.VTK_TRIANGLE)
    3
    """
    # Get the VTK class name for the cell type
    classname = vtk.vtkCellTypes().GetClassNameFromTypeId(cell_type)

    # Get the actual VTK class
    vtk_class = getattr(vtk, classname)

    # Create an instance of the cell
    cell = vtk_class()

    # Get number of points
    number_of_points = cell.GetNumberOfPoints()

    if number_of_points <= 0:
        msg = f"Cell type {classname} ({cell_type}) is variable-sized (only fixed-sized is supported)."
        raise AttributeError(msg)

    return number_of_points


def construct_cells(connectivity: list, cell_types: list) -> np.array:
    """
    Construct a VTK cells array from connectivity + cell types.
    Only supports fixed-size cell types.

    Parameters
    ----------
    connectivity : Concatenated point indices for all cells. VTK convention.
    cell_types :   VTK cell types (same length as number of cells).

    Returns
    -------
    cells : Flattened VTK-style cell array [npts, id0, id1, ...]. Pyvista convention
    """
    connectivity = np.asarray(connectivity, dtype=int)
    cell_types = np.asarray(cell_types, dtype=np.uint8)

    cells = []
    offset = 0
    for ctype in cell_types:
        npts = cell_points(ctype)
        ids = connectivity[offset : offset + npts]
        offset += npts
        cells.extend([npts, *ids])

    return np.array(cells, dtype=int)
