from pathlib import Path

import numpy as np
import pandamesh as pm
from geopandas import GeoDataFrame, read_file


def _prepare_shp_for_meshing(shape_file: str | Path) -> GeoDataFrame:
    """
    This function is to prepare the shapefile for meshing with pandamesh.
    Therefore it is read with geopands as a GeoDataFrame. The GeoDataFrame is
    prepared for meshing.

    :param shape_file: Path of shape-file to be prepared for meshing.
    :return: GeoDataFrame ready to get meshed.
    """
    shape_file = Path(shape_file)
    gdf = read_file(shape_file)
    if "MultiPolygon" in gdf["geometry"].geom_type.to_numpy():
        # break down multipolygon to multiple polygons
        exploded_data = gdf.explode()
        gdf = GeoDataFrame(geometry=list(exploded_data["geometry"]))
    # get rid off intersections
    union = gdf.union_all()
    union_gdf = GeoDataFrame(geometry=[union])
    # breakdown multipolygon again
    exploded_union = union_gdf.explode()
    return GeoDataFrame(geometry=list(exploded_union["geometry"]))


def shapefile_meshing(
    shapefile: str | Path,
    simplify: bool = False,
    triangle: bool = True,
    cellsize: int | None = None,
) -> tuple:
    """
    Generate a triangle- or GMSH-mesh from a shapefile.

    :param shapefile: Shapefile to be meshed.
    :param simplify: With the Douglas-Peucker algorithm the geometry is simplified. The original line
        is split into smaller parts. All points with a distance smaller than half the cellsize are removed.
        Endpoints are preserved. More infos at https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoSeries.simplify.html.
    :param triangle: Use the triangle-mesher. If False, the gmsh-mesher is used.
    :param cellsize: Size of the cells in the mesh.

    :return: tuple of points and cells of the mesh.
    """
    geodataframe = _prepare_shp_for_meshing(shapefile)
    if cellsize is None:
        bounds = geodataframe.total_bounds
        x_length = bounds[2] - bounds[0]
        y_length = bounds[3] - bounds[1]
        cellsize = x_length / 100 if x_length > y_length else y_length / 100
    if simplify:
        geodataframe.geometry = geodataframe.geometry.buffer(cellsize / 20)
        geodataframe["dissolve_column"] = 0
        geodataframe = geodataframe.dissolve(by="dissolve_column")
        geodataframe.geometry = geodataframe.geometry.simplify(cellsize / 2)
        exploded_union = geodataframe.explode()
        geodataframe = GeoDataFrame(geometry=list(exploded_union["geometry"]))
    geodataframe["cellsize"] = cellsize

    if triangle:
        mesher = pm.TriangleMesher(geodataframe)
    else:
        mesher = pm.GmshMesher(geodataframe)
    return mesher.generate()


def mesh_from_points_cells(
    points: np.ndarray, cells: np.ndarray
) -> pv.UnstructuredGrid:
    """
    Create a PyVista UnstructuredGrid from points and cells. Pyvista requires point, cell and celltype
    array to set up a unstructured grid. This function creates the cell array and cell type array in the
    correct structure automatically. So, simply an array of points with coordinates and an array of cells
    with indices of points are needed. For more information see:
    https://docs.pyvista.org/version/stable/api/core/_autosummary/pyvista.unstructuredgrid
    :param points: An array of shape (n_points, 3) containing the coordinates of each point.
    :param cells: An array of lists, where each inner list represents a cell and contains the indices of its points.
    :return: A Mesh object
    """
    # Convert points to numpy array if it's not already
    points = np.asarray(points)
    cells = cells.astype(np.int32, copy=False)
    # Append the zeros column to the points, if they refer to 2D data.
    if points.shape[1] == 2:
        zeros_column = np.zeros((points.shape[0], 1), dtype=int)
        points = np.column_stack((points, zeros_column))
    # Prepare the cell array
    cell_array = np.concatenate([np.r_[len(cell), cell] for cell in cells])
    assert (
        len(np.unique([len(cell) for cell in cells])) == 1
    ), "All cells must be of the same type. Hence, have the same number of points. If not use pyvista.UnstructuredGrid directly."
    # choose the celltype:
    if cell_array[0] == 4:
        celltype = pv.CellType.TETRA
    elif cell_array[0] == 8:
        celltype = pv.CellType.HEXAHEDRON
    elif cell_array[0] == 6:
        celltype = pv.CellType.WEDGE
    elif cell_array[0] == 5:
        celltype = pv.CellType.PYRAMID
    elif cell_array[0] == 3:
        celltype = pv.CellType.TRIANGLE
    elif cell_array[0] == 2:
        celltype = pv.CellType.LINE
    else:
        celltype = pv.CellType.CONVEX_POINT_SET
    # Create the cell types array
    cell_types = np.full(len(cells), celltype)
    # Return the UnstructuredGrid
    return pv.UnstructuredGrid(np.asarray(cell_array), cell_types, points)
