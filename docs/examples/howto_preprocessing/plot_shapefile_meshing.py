r"""
Meshing a shapefile
===================

.. sectionauthor:: Julian Heinze (Helmholtz Centre for Environmental Research GmbH - UFZ)

In this example we present how to create a unstructured grid in vtk format (\*.vtu) from a shapefile.
"""

# %%
# Necessary imports
import ogstools.meshlib as ml
from ogstools.examples import test_shapefile

# %%
# A shapefile can be directly read with meshlib to create a mesh.
mesh = ml.Mesh.read(test_shapefile)
mesh.plot(show_edges=True)
