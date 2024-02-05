"""
Aspect ratios
=============

.. sectionauthor:: Florian Zill (Helmholtz Centre for Environmental Research GmbH - UFZ)

By default plots with meshplotlib try to retain the true mesh proportions.
If the meshes aspect ratio lies outside of predefined limits
(setup.min_ax_aspect, setup.max_ax_aspect) the axes get compressed to stay
inside the given limits. The following examples shall illustrate this
behaviour.
"""

# %%
import numpy as np
import pyvista as pv

from ogstools.meshplotlib import plot, setup

setup.reset()
print(f"{setup.min_ax_aspect=}")
print(f"{setup.max_ax_aspect=}")


# sphinx_gallery_start_ignore
def custom_mesh(dx: float, dy: float):
    n = 50
    x = np.linspace(0, dx, num=n)
    y = np.linspace(0, dy, num=n)
    xx, yy = np.meshgrid(x, y)
    zz = np.zeros(xx.shape)
    points = np.c_[xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)]
    mesh = pv.PolyData(points).delaunay_2d()
    mesh.point_data["example"] = np.sin(mesh.points[:, 0]) * np.cos(
        mesh.points[:, 1]
    )
    return mesh


# sphinx_gallery_end_ignore


# %%
fig = plot(custom_mesh(np.pi * 2, np.pi), "example")
# %%
fig = plot(custom_mesh(np.pi * 4, np.pi), "example")
# %%
fig = plot(
    [custom_mesh(np.pi * 2, np.pi), custom_mesh(np.pi * 2, np.pi)], "example"
)
# %%
fig = plot(custom_mesh(np.pi, np.pi * 2), "example")
# %%
fig = plot(custom_mesh(np.pi, np.pi * 4), "example")
# %%
fig = plot(
    [custom_mesh(np.pi, np.pi * 2), custom_mesh(np.pi, np.pi * 2)], "example"
)

# %% [markdown]
# You can enforce true proportions regardless of the resulting figures
# dimensions, by setting the limiting values to None.

# %%
setup.min_ax_aspect = None
setup.max_ax_aspect = None
fig = plot(custom_mesh(np.pi * 3, np.pi), "example")

# %%
fig = plot(custom_mesh(np.pi, np.pi * 3), "example")
