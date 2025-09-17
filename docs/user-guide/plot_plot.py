"""
plotting model data
===================

One strength of ``ogstools`` is to simplify the process of plotting mesh data.

For 2D meshes (or slices of 3D meshes) we use the ``matplotlib`` backend, as it
is able to generate paper-quality and highly customizable figures.

For 3D meshes we use the `pyvista backend <https://docs.pyvista.org/examples/02-plot/index.html>`_,
as it is better suited for this task.

Here we show basic usage. More advanced usage can be found in the :doc:`examples </auto_examples/index>`

We use ``Variables`` to specify which data to plot,
(:ref:`sphx_glr_auto_examples_howto_postprocessing_plot_variables.py`)
but strings as an argument work equally well.
"""

# %%

# sphinx_gallery_start_ignore

# sphinx_gallery_thumbnail_number = 3

# sphinx_gallery_end_ignore

import ogstools as ot
from ogstools import examples

mesh_2D = examples.load_meshseries_THM_2D_PVD()[-1]
mesh_3D = examples.load_meshseries_diffusion_3D()[-1]

# %% [markdown]
# Plotting filled 2D contours returns a ``matplotlib`` figure. Via `fig.axes`
# you have access to the `Axes` object and can further customize and plot to
# your liking.

# %%
fig = mesh_2D.plot_contourf(ot.variables.material_id)

# %% [markdown]
# Plotting 3D meshes returns a ``pyvista.Plotter`` object, which you can add
# other meshes to or change some visualization parameters. Note: In the
# interactive scene the colorbar labels may be faulty until you start moving the
# model around.

# %%
plotter = mesh_3D.plot_contourf(ot.variables.temperature)
plotter.show()

# %% [markdown]
# Plotting cell data also works. In this case we plot categorical material ids
# and have some of them labeled. You can also add a dictionary to map
# transparencies to the individual ids. Passing ``interactive=false`` will
# return a screenshot instead of a ``Plotter``.

# %%
# adding a synthetic MaterialIDs field for demonstration purposes
cpts = mesh_3D.cell_centers().points
mesh_3D.cell_data["MaterialIDs"] = (12 * cpts[:, 0] + 3).astype(int)

# %%
#
# .. code-block:: python
#
#    mesh_3D.plot_contourf(
#        "MaterialIDs", opacities={7: 0.1, 10: 0.9}, interactive=False
#    )
#

# You have to be aware, if you switch between local and remote rendering in your
# notebook, that there are some differences in the resulting figure. E.g. remote
# rendering seems to have trouble correctly labeling categorical values.
# sphinx_gallery_start_ignore
mesh_3D.plot_contourf("MaterialIDs", opacities={7: 0.1, 10: 0.9}).screenshot(
    return_img=False
)
# sphinx_gallery_end_ignore


# %% [markdown]
# When passing ``interactive=true``, also 2D meshes will use the ``pyvista``
# backend:

plotter = mesh_2D.plot_contourf(ot.variables.temperature, interactive=True)
plotter.show()
