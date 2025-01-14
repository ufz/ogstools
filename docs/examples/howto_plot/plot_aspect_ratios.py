"""
Aspect ratios
=============

.. sectionauthor:: Florian Zill (Helmholtz Centre for Environmental Research GmbH - UFZ)

By default, filled contour plots try to retain the true mesh proportions.
If the meshes aspect ratio lies outside of predefined limits
(setup.min_ax_aspect, setup.max_ax_aspect) the axes get compressed to stay
inside the given limits. The following examples shall illustrate this
behaviour.
"""

# %%
import numpy as np
import pyvista as pv

import ogstools as ot

print(f"{ot.plot.setup.min_ax_aspect=}")
print(f"{ot.plot.setup.max_ax_aspect=}")


# sphinx_gallery_start_ignore
# TODO: move to examples
def custom_mesh(dx: float, dy: float):
    number_of_points = 50
    x = np.linspace(0, dx, num=number_of_points)
    y = np.linspace(0, dy, num=number_of_points)
    xx, yy = np.meshgrid(x, y)
    zz = np.zeros(xx.shape)
    points = np.c_[xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)]
    mesh = pv.PolyData(points).delaunay_2d()
    mesh.point_data["example"] = np.sin(mesh.points[:, 0]) * np.cos(
        mesh.points[:, 1]
    )
    return mesh


# sphinx_gallery_end_ignore

# %% [markdown]
# The following fits inside the defined limits and gets displayed with true
# proportions.

# %%
fig = ot.plot.contourf(custom_mesh(np.pi * 2, np.pi), "example")
# %% [markdown]
# This one would be too wide and thus and gets compressed to fit the maximum
# aspect ratio.

# %%
fig = ot.plot.contourf(custom_mesh(np.pi * 4, np.pi), "example")
# %% [markdown]
# When plotting multiple meshes together, this applies to each subplot.
# So here each subplot has true proportions again since each one fits the limits.

# %%
fig = ot.plot.contourf(
    [custom_mesh(np.pi * 2, np.pi), custom_mesh(np.pi * 2, np.pi)], "example"
)
# %% [markdown]
# The following figure would be to tall and is clipped to the minimum aspect
# ratio.

# %%
fig = ot.plot.contourf(custom_mesh(np.pi, np.pi * 3), "example")
# %% [markdown]
# The same is true here:

# %%
fig = ot.plot.contourf(
    [custom_mesh(np.pi, np.pi * 3), custom_mesh(np.pi, np.pi * 3)], "example"
)

# %% [markdown]
# You can enforce true proportions regardless of the resulting figures
# dimensions, by setting the limiting values to None. In this case we get a
# very wide figure.

# %%
ot.plot.setup.min_ax_aspect = None
ot.plot.setup.max_ax_aspect = None
fig = ot.plot.contourf(custom_mesh(np.pi * 3, np.pi), "example")

# %% [markdown]
# And in this case we get a very tall figure.

# %%
fig = ot.plot.contourf(custom_mesh(np.pi, np.pi * 3), "example")
