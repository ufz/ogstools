from pathlib import Path

import numpy as np
from geopandas import GeoDataFrame, read_file
from pyvista import UnstructuredGrid

# import pandamesh as pm
# import meshio


def prepare_shp_for_meshing(shape_file: Path) -> GeoDataFrame:
    """
    This function is to prepare the shapefile for meshing with pandamesh.
    Therefore it is read with geopands as a GeoDataFrame. The GeoDataFrame is
    prepared for meshing.

    some params...
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


shp = Path("simplified_land_polygons.shp")
print(prepare_shp_for_meshing(str(shp))[0])
"""
def geodataframe_meshing(case):
    # define cellsize
    final_gdf["cellsize"] = args.cellsize
    if args.simplify == "simplified":
        # simplified = final_gdf.copy()
        final_gdf.geometry = final_gdf.geometry.buffer(args.cellsize / 20)
        final_gdf["dissolve_column"] = 0
        final_gdf = final_gdf.dissolve(by="dissolve_column")
        final_gdf.geometry = final_gdf.geometry.simplify(args.cellsize / 2)
        exploded_union = final_gdf.explode()
        # create final geodataframe
        final_gdf = gp.GeoDataFrame(
            geometry=[polygon for polygon in exploded_union["geometry"]]
        )
        final_gdf["cellsize"] = args.cellsize

    # choose the meshing algorithm: also GmshMesher() possible, but requires installation of gmsh or triangle
    if args.meshing == "Triangle":
        mesher = pm.TriangleMesher(final_gdf)
    elif args.meshing == "GMSH":
        mesher = pm.GmshMesher(final_gdf)
    return mesher.generate()
"""


def _create_pyvista_ugrid(points, cells) -> UnstructuredGrid:
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

    # Prepare the cell array
    cell_array = []
    for cell in cells:
        cell_array.append(len(cell))
        cell_array.extend(cell)

    # Create the cell types array
    cell_types = np.full(len(cells), pv.CellType.POLYGON)

    # Create the UnstructuredGrid
    ugrid = UnstructuredGrid(np.array(cell_array), cell_types, points)

    return ugrid


# mesh.write(args.output)
