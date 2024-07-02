from pathlib import Path

import numpy as np
from geopandas import GeoDataFrame, read_file
from pyvista import UnstructuredGrid

import pandamesh as pm
import meshio
import pyvista as pv


def prepare_shp_for_meshing(shape_file: Path) -> GeoDataFrame:
    """
    This function is to prepare the shapefile for meshing with pandamesh.
    Therefore it is read with geopands as a GeoDataFrame. The GeoDataFrame is
    prepared for meshing.

    :param shape_file: Path of shape-file to be prepared for meshing.
    """
    gdf = read_file(shape_file)
    if "MultiPolygon" in gdf["geometry"].geom_type.values:
        # break down multipolygon to multiple polygons
        exploded_data = gdf.explode()
        # from geoseries to geodataframe
        gdf = GeoDataFrame(
            geometry=[polygon for polygon in exploded_data["geometry"]]
        )
    # get rid off intersections
    union = gdf.union_all()
    # union to geodataframe
    union_gdf = GeoDataFrame(geometry=[union])
    # breakdown multipolygon again
    exploded_union = union_gdf.explode()
    # create final geodataframe
    final_gdf = GeoDataFrame(
        geometry=[polygon for polygon in exploded_union["geometry"]]
    )
    return final_gdf


def geodataframe_meshing(
    geodataframe: GeoDataFrame,
    simplify: bool = False,
    triangle: bool = True,
    cellsize: int | None = None,
) -> tuple:
    """
    Generate a triangle- or GMSH-mesh from a shapefile read as GeoDataFrame.
    """
    if simplify:
        geodataframe.geometry = geodataframe.geometry.buffer(cellsize / 20)
        geodataframe["dissolve_column"] = 0
        geodataframe = geodataframe.dissolve(by="dissolve_column")
        geodataframe.geometry = geodataframe.geometry.simplify(cellsize / 2)
        exploded_union = geodataframe.explode()
        # create final geodataframe
        geodataframe = GeoDataFrame(
            geometry=[polygon for polygon in exploded_union["geometry"]]
        )
    if cellsize == None:
        bounds = geodataframe.total_bounds
        x_length = bounds[2] - bounds[0]
        y_length = bounds[3] - bounds[1]
        cellsize = x_length / 100 if x_length > y_length else y_length / 100
    geodataframe["cellsize"] = cellsize
    # choose the meshing algorithm: also GmshMesher() possible,
    # but requires installation of gmsh or triangle
    if triangle:
        mesher = pm.TriangleMesher(geodataframe)
    else:
        # ToDo: GmshMesher does not supply correct structure of point and
        # cell data to create pyvista UnstructuredGrid.
        mesher = pm.GmshMesher(geodataframe)
    return mesher.generate()


def create_pyvista_mesh(points, cells) -> UnstructuredGrid:
    """
    Create a PyVista UnstructuredGrid from points and cells.

    Parameters:
    points (np.array): An array of shape (n_points, 3) containing the coordinates of each point.
    cells (list): A list of lists, where each inner list represents a cell and contains the indices of its points.

    Returns:
    pv.UnstructuredGrid: The created PyVista UnstructuredGrid object.
    """
    # Convert points to numpy array if it's not already
    points = np.array(points)

    # Append the zeros column to the points, if they refer to 2D data.
    if points.shape[1] == 2:
        zeros_column = np.zeros((points.shape[0], 1), dtype=int)
        points = np.column_stack((points, zeros_column))
    # Prepare the cell array
    cell_array = []
    for cell in cells:
        cell_array.append(len(cell))
        cell_array.extend(cell)

    # Create the cell types array
    cell_types = np.full(len(cells), pv.CellType.POLYGON)

    # Create the UnstructuredGrid
    mesh = UnstructuredGrid(np.array(cell_array), cell_types, points)

    return mesh
