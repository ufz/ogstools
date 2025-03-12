from pathlib import Path

import pyvista as pv


def read_shape(
    shapefile: str | Path,
    simplify: bool = False,
    cellsize: int | None = None,
) -> pv.UnstructuredGrid:
    """
    Generate a pyvista Unstructured grid from a shapefile.

    :param shapefile: Shapefile to be meshed.
    :param simplify: With the Douglas-Peucker algorithm the geometry is simplified. The original line
        is split into smaller parts. All points with a distance smaller than half the cellsize are removed.
        Endpoints are preserved. More infos at https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoSeries.simplify.html.
    :param cellsize: Size of the cells in the mesh - only needed for simplify algorithm.
        If None - cellsize is 1/100 of larger bound (x or y).

    :returns: pv.UnstructuredGrid
    """

    from ogstools.meshlib.shape_meshing_utils import (
        _mesh_from_points_cells,
        _points_cells_from_shapefile,
    )

    points_cells = _points_cells_from_shapefile(shapefile, simplify, cellsize)
    return _mesh_from_points_cells(points_cells[0], points_cells[1])
