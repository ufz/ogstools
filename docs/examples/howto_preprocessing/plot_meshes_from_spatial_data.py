"""
Creating Meshes from spatial data
=================================

This example shows how to mesh geodata from files or GeoDataFrames with :meth:`~ogstools.Meshes.from_spatial_data`.
"""

# sphinx_gallery_start_ignore

# sphinx_gallery_thumbnail_number = 1

# sphinx_gallery_end_ignore

# %%
import geopandas as gpd
import numpy as np
import pyvista as pv
import shapely
from matplotlib import pyplot as plt

import ogstools as ot
from ogstools import examples

ot.plot.setup.linewidth = 0.6
ot.plot.setup.fontsize = 20


def plot_height_profile(points: pv.pyvista_ndarray) -> plt.Figure:
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        cmap="terrain",
        edgecolor="none",
        alpha=0.9,
    )
    ax.view_init(elev=30, azim=-60)
    ax.set_box_aspect((1, 1, 0.5))
    return fig


# %% [markdown]
# This function can be used to generate :class:`~ogstools.Meshes` objects from geodata given either in files or as GeoDataFrames.
# It will create consistent submeshes for each file/layer you put into it.
# Maximum cellsizes can be specified globally, attached as columns to the input data or calculated dynamically via a gmsh size callback or a gmsh MathEval field.

# %% [markdown]
# Basic usage
# -----------
# To generate a basic Meshes object, simply call :meth:`~ogstools.Meshes.from_spatial_data` with your file (or list of files) or a GeoDataFrame, a maximum cellsize and the name of the domain geometries.
# If meshing fails or produces unexpected results, you can pass `log=True` to get the gmsh log printed during runtime.

# %%
meshes = ot.Meshes.from_spatial_data(
    examples.selke_complete_gpkg,
    cellsize=2000,
    domain="Selke_Basin_complete_Domain_homo",
)
fig = meshes.plot(lw=0.6)
list(meshes)

# %% [markdown]
# Geodata inputs
# --------------
# You can input geodata from .shp, .gpkg and .geojson files like we have done above.
# These can be downloaded from various repositories or built with a GIS tool like `QGIS <https://docs.qgis.org/latest/en/docs/training_manual/create_vector_data/index.html>`_.
# Alternatively, you can input geodata as a GeoDataFrame from geopandas if you have the data in some other geopandas-compatible format.
# Additionally, this allows you to construct the geodata to be meshed in Python using shapely:

# %%
square = shapely.Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
gdf_square = gpd.GeoDataFrame(geometry=[square])
meshes_square = ot.Meshes.from_spatial_data(
    gdf_square, cellsize=2, domain="frame_1"
)
fig_square = meshes_square.plot(lw=0.6)

# %% [markdown]
# .. admonition:: Capabilities and limitations
#    :class: note
#
#    * Only surfaces can be meshed (though lines and points may be created as submeshes).
#    * Only Points, MultiPoints, LineStrings, MultiLineStrings, Polygons and MultiPolygons are supported. Any other geometry is ignored.
#    * All files/GeoDataFrames must have the same CRS attached.
#    * Polygons with holes are accepted.
#    * Polygons may overlap.
#    * Lines may only intersect each other or polygon boundaries if both features have a vertex at the intersection.


# %% [markdown]
# Domain and submeshes
# --------------------
# A full Meshes object has a domain and consistent submeshes, not only one simple mesh of one or more polygons.
# To get one, give the input data in separate parts.
# To do this, pass a list of the above arguments (files, GeoDataFrames).
# Alternatively, pass geopackage files with multiple layers, which give one submesh per layer.
# You can also specify the name of the file, layer or frame that contains the domain.
# If you do so, the domain file, layer or frame may only contain polygons.
# No submesh is created for the domain.
# If you do not specify a domain, the domain is simply assumed to be the union of all polygons in all inputs.
# Every feature is clipped to the domain, so that the result is consistent.
# This is either the domain you specified or the union of all polygons.


# %%
line = shapely.LineString([(1, 1), (9, 9)])
gdf_line = gpd.GeoDataFrame(geometry=[line])
meshes_square_line = ot.Meshes.from_spatial_data(
    [gdf_square, gdf_line], domain="frame_1", cellsize=2
)
fig_square_line = meshes_square_line.plot(lw=0.6)

# %% [markdown]
# A note on naming
# ----------------
# The names of the domain and submeshes are generated from the input geodata:
# For .shp and .geojson files the mesh name will be the filename.
# For layers in .gpkg files the mesh name will be filename_layername.
# GeoDataFrames will be numbered as frame_1, frame_2, etc.
# Alternatively, you can input tuples of strings and GeoDataFrames for which the string will be chosen as name.

# %%
meshes_with_name = ot.Meshes.from_spatial_data(
    [("Square Domain", gdf_square), ("Diagonal", gdf_line)],
    domain="Square Domain",
    cellsize=2,
)
fig_with_name = meshes_with_name.plot(lw=0.6)

# %% [markdown]
# .. note::
#    Whenever two submeshes would have the same name using this system the first is simply overwritten by the second.
#    Additionally, specifying a non-existent mesh name as domain will cause the function to raise an error.

# %% [markdown]
# Cellsizes
# ---------
# The `cellsize` parameter specifies a global maximum cellsize for the resulting mesh.
# To specify different maximum cellsizes at different parts of the mesh, you have two options:
# You add a cellsize column to the input geodata or pass a function for gmsh to dynamically calculate cellsizes.
# To add a column to the geodata you can use a GIS tool like QGIS.
#
# .. figure:: /examples/howto_preprocessing/two_squares_cellsizes.png
#    :alt: Screenshot of QGIS showing the attribute table with a cellsize column
#
#    Screenshot of QGIS showing a simple polygon layer of a geopackage and the corresponding attribute table with cellsizes.


# %% [markdown]
# You can also add a cellsize column to a GeoDataFrame using geopandas.

# %%
square_1 = shapely.Polygon([(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)])
square_2 = shapely.Polygon(
    [(100, 0), (200, 0), (200, 100), (100, 100), (100, 0)]
)
gdf_two_squares = gpd.GeoDataFrame(
    {"cellsize": [10, 25]}, geometry=[square_1, square_2]
)
meshes_two_squares = ot.Meshes.from_spatial_data(
    ("domain", gdf_two_squares), domain="domain"
)
fig_two_squares = meshes_two_squares.plot(lw=0.6)

# %% [markdown]
# To add a function for gmsh, use the parameter `cellsize_func`.
# You can pass it either a string or a Callable (i.e. a Python function).
# A string is evaluated as a gmsh `MathEval <https://gmsh.info/doc/texinfo/#index-MathEval>`_ field.
# A Callable is evaluated as a gmsh `size callback <https://gmsh.info/doc/texinfo/gmsh.html#index-gmsh_002fmodel_002fmesh_002fsetSizeCallback>`_.
# In both cases, it is passed directly to gmsh and must therefore conform to the format gmsh requires (see the respective links above).


# %%
def periodic_from_center(
    dim: int, tag: int, x: float, y: float, z: float, lc: float  # noqa: ARG001
) -> float:
    center = (5, 5)
    scaling = 3
    floor = 0.1
    distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    return floor + scaling * abs(np.sin(np.pi * distance))


center = shapely.Point((5, 5))
gdf_center = gpd.GeoDataFrame(geometry=[center])
meshes_periodic = ot.Meshes.from_spatial_data(
    [("square", gdf_square), ("middle", gdf_center)],
    domain="square",
    cellsize_func=periodic_from_center,
)
fig_periodic = meshes_periodic.plot(lw=0.6)

# %% [markdown]
# .. note::
#    If you do not specify cellsizes in any way, the global maximum cellsize is calculated as a hundredth of the square root of the domain area.


# %% [markdown]
# 3D coordinates and height interpolation
# ---------------------------------------
# Since meshing surfaces with 3D coordinates is very computationally expensive, all coordinates are projected to the xy-plane before they are added to gmsh.
# This might lead to unexpected results or crashes if the projection is not injective (e.g. because the domain has an overhang).
# After successful meshing the z-coordinates from the data are interpolated onto the result.

# %%
square_3D = shapely.Polygon(
    [(0, 0, 1), (10, 0, 0), (10, 10, 1), (0, 10, 2), (0, 0, 1)]
)
gdf_square_3D = gpd.GeoDataFrame(geometry=[square_3D])
meshes_square_3D = ot.Meshes.from_spatial_data(
    gdf_square_3D, cellsize=2, domain="frame_1"
)
fig_square_3D = meshes_square_3D.plot(lw=0.6)
fig_square_height = plot_height_profile(meshes_square_3D.domain.points)

# %% [markdown]
# Additionally, you can pass your own height_data as an array of points with 3 coordinates.
# These points are not forced into the mesh, but their z-values are used in the interpolation.

# %%
additional_heights = np.array([[5, 5, 2]])
meshes_square_add_height = ot.Meshes.from_spatial_data(
    gdf_square_3D, cellsize=2, domain="frame_1", height_data=additional_heights
)
fig_square_add_height = plot_height_profile(
    meshes_square_add_height.domain.points
)
# %% [markdown]
# .. note::
#    Height interpolation both of input heights as well as additional height_data distorts cells and might therefore result in cellsizes larger than specified.
