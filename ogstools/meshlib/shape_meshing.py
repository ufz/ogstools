from pathlib import Path

import numpy as np
import pandamesh as pm
import pyvista as pv
from geopandas import GeoDataFrame, read_file


def _prepare_shp_for_meshing(shape_file: str | Path) -> GeoDataFrame:
    """
    This function is to prepare the shapefile for meshing with pandamesh.
    Therefore it is read with geopands as a GeoDataFrame. The GeoDataFrame is
    prepared for meshing.

    :param shape_file: Path of shape-file to be prepared for meshing.
    :returns: GeoDataFrame ready to get meshed.
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


def _points_cells_from_shapefile(
    shapefile: str | Path,
    simplify: bool = False,
    mesh_generator: str = "triangle",
    cellsize: int | None = None,
) -> tuple:
    """
    Generate a triangle- or GMSH-mesh from a shapefile.

    :param shapefile: Shapefile to be meshed.
    :param simplify: With the Douglas-Peucker algorithm the geometry is simplified. The original line
        is split into smaller parts. All points with a distance smaller than half the cellsize are removed.
        Endpoints are preserved. More infos at https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoSeries.simplify.html.
    :param triangle: Use the triangle-mesher. If False, the gmsh-mesher is used.
    :param cellsize: Size of the cells in the mesh - only needed for simplify algorithm.
        If None - cellsize is 1/100 of larger bound (x or y).

    :returns: tuple of points and cells of the mesh.
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

    assert mesh_generator in [
        "triangle",
        "gmsh",
    ], "mesh_generator must be 'triangle' or 'gmsh'"

    if mesh_generator == "triangle":
        mesher = pm.TriangleMesher(geodataframe)
        points_cells = mesher.generate()
    elif mesh_generator == "gmsh":
        mesher = pm.GmshMesher(geodataframe)
        points_cells = mesher.generate()
        pm.GmshMesher.finalize()

    return points_cells


def _mesh_from_points_cells(
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

    :returns: A Mesh object
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
    celltype_mapping = {
        4: pv.CellType.TETRA,
        8: pv.CellType.HEXAHEDRON,
        6: pv.CellType.WEDGE,
        5: pv.CellType.PYRAMID,
        3: pv.CellType.TRIANGLE,
        2: pv.CellType.LINE,
    }

    celltype = celltype_mapping.get(cell_array[0], pv.CellType.CONVEX_POINT_SET)
    cell_types = np.full(len(cells), celltype)
    # Return the UnstructuredGrid
    return pv.UnstructuredGrid(np.asarray(cell_array), cell_types, points)


def read_shape(
    shapefile: str | Path,
    simplify: bool = False,
    mesh_generator: str = "triangle",
    cellsize: int | None = None,
) -> pv.UnstructuredGrid:
    """
    Generate a pyvista Unstructured grid from a shapefile.

    :param shapefile: Shapefile to be meshed.
    :param simplify: With the Douglas-Peucker algorithm the geometry is simplified. The original line
        is split into smaller parts. All points with a distance smaller than half the cellsize are removed.
        Endpoints are preserved. More infos at https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoSeries.simplify.html.
    :param mesh_generator: Choose between 'triangle' and 'gmsh' to generate the mesh.
    :param cellsize: Size of the cells in the mesh - only needed for simplify algorithm.
        If None - cellsize is 1/100 of larger bound (x or y).

    :returns: pv.UnstructuredGrid
    """
    points_cells = _points_cells_from_shapefile(
        shapefile, simplify, mesh_generator, cellsize
    )
    return _mesh_from_points_cells(points_cells[0], points_cells[1])
