from pathlib import Path

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
