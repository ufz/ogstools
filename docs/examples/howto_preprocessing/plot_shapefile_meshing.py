r"""
Meshing a shapefile
===================

.. sectionauthor:: Julian Heinze (Helmholtz Centre for Environmental Research GmbH - UFZ)

In this example we present how to create a unstructured grid in vtk format (\*.vtu) from a shapefile.
"""

# %%
# 0. Necessary imports
from ogstools.examples import test_shapefile
from ogstools.meshlib import (
    create_pyvista_mesh,
    geodataframe_meshing,
    prepare_shp_for_meshing,
)

# %%
# 1. Prepare the shapefile for meshing
geodataframe = prepare_shp_for_meshing(test_shapefile)
# %%
# 2. Mesh the geodataframe
points_cells = geodataframe_meshing(geodataframe)
pyvista_mesh = create_pyvista_mesh(
    points=points_cells[0], cells=points_cells[1]
)
# %%
# 3. Visualize the data.
pyvista_mesh.plot(show_edges=True)
