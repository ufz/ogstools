"""
Extract a 1D profile from 2D and plot it
========================================

.. sectionauthor:: Feliks Kiszkurno (Helmholtz Centre for Environmental Research GmbH - UFZ)
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

import ogstools as ogs
from ogstools import examples

# %% [markdown]
# Single fracture
# ------------------
# Define a profile line by providing a list of points in x, y, z coordinates
# and load an example data set:

# %%
mesh = examples.load_meshseries_HT_2D_XDMF().mesh(-1)

profile_HT = np.array([[4, 2, 0], [4, 18, 0]])

# %%
mesh_sp, mesh_kp = ogs.meshlib.sample_polyline(
    mesh, ["pressure", "temperature"], profile_HT
)

# %% [markdown]
# It has returned a pandas DataFrame containing all information about the
# profile and a numpy array with the position of the "knot-points".
# Let's investigate the DataFrame first:

# %%
mesh_sp.head(10)

# %% [markdown]
# We can see the spatial coordinates of points on the profile ("x", "y", "z"
# - columns), distances from the beginning of the profile ("dist") and within
# current segment ("dist_in_segment"). Note, that since we defined our profile
# on only two points, there is only one segment, hence in this special case
# columns dist and dist_in_segment are identical. At the end of the DataFrame
# we can can find two columns with the variables that we are interested in:
# "temperature" and "pressure". Each occupies one column, as those are scalar
# values. Using columns "dist", "pressure" and "temperature" we can easily
# plot the data:

# %%
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax = mesh.plot_linesample(
    x="dist",
    variable="pressure",
    profile_points=profile_HT,
    ax=ax,
    fontsize=15,
)
ax_twinx = ax.twinx()
ax_twinx = mesh.plot_linesample(
    x="dist",
    variable="temperature",
    profile_points=profile_HT,
    ax=ax_twinx,
    fontsize=15,
)
ogs.plot.utils.color_twin_axes(
    [ax, ax_twinx],
    [ogs.variables.pressure.color, ogs.variables.temperature.color],
)
fig.tight_layout()


# %% [markdown]
# What happens when we are interested in a vector variable?
# We can see it in the following example using the Darcy velocity:

# %%
mesh_sp, mesh_kp = ogs.meshlib.sample_polyline(
    mesh, "darcy_velocity", profile_HT
)

# %%
mesh_sp.head(5)

# %% [markdown]
# Now we have two columns for the variable. The Darcy velocity is a vector,
# therefore "sample_over_polyline" has split it into two columns and appended
# the variable name with increasing integer. Note, that this suffix has no
# physical meaning and only indicates order. It is up to user to interpret it
# in a meaningful way. By the
# `OpenGeoSys conventions <https://www.opengeosys.org/docs/userguide/basics/conventions/#a-namesymmetric-tensorsa--symmetric-tensors-and-kelvin-mapping>`_,
# "darcy_velocity_0" will be in the x-direction and "darcy_velocity_1" in
# y-direction.


# %% [markdown]
# Elder benchmark
# ------------------
# In this example we will use a Variable object from the ogstools to
# sample the data. This allows "sample_over_polyline" to automatically
# convert from the "data_unit" to the "output_unit":

# %%
profile_CT = np.array([[47.0, 1.17, 72.0], [-4.5, 1.17, -59.0]])
mesh = examples.load_meshseries_CT_2D_XDMF().mesh(11)

# %%
mesh_sp, mesh_kp = ogs.meshlib.sample_polyline(
    mesh, ogs.variables.saturation, profile_CT
)

# %% [markdown]
# As before we can see the profile parameters and propertiy values in a
# DataFrame:

# %%
mesh_sp.head(5)


# %% [markdown]
# This time we will prepare more complicated plot showing both the mesh and
# the profile.

# %%
fig, ax = mesh.plot_linesample_contourf(
    ogs.variables.saturation, profile_CT, resolution=100
)

# %% [markdown]
# THM
# ------
# It is also possible to obtain more than one variable at the same time using
# more complex profiles. They can be constructed by providing more than
# 2 points. With those points:

# %%
profile_THM = np.array(
    [
        [-1000.0, -175.0, 6700.0],  # Point A
        [-600.0, -600.0, 6700.0],  # Point B
        [100.0, -300.0, 6700.0],  # Point C
        [3500, -900.0, 6700.0],  # Point D
    ]
)
# %% [markdown]
# the profile will run as follows:
#
# .. math::
#
#   \text{AB} \rightarrow \text{BC} \rightarrow \text{CD}
#
# Point B will at the same time be the last point in the first segment AB
# and first one in second segment BC, however in the returned array,
# it will occur only once.
# For this example we will use a different dataset:

# %%
mesh = examples.load_meshseries_THM_2D_PVD().mesh(-1)

# %%
ms_THM_sp, dist_at_knot = ogs.meshlib.sample_polyline(
    mesh,
    [ogs.variables.pressure, ogs.variables.temperature],
    profile_THM,
    resolution=100,
)

# %% [markdown]
# Again, we can investigate the returned DataFrame, but this time we will
# have a look at its beginning:

# %%
ms_THM_sp.head(5)

# %%
# and end:

# %%
ms_THM_sp.tail(10)

# %% [markdown]
# Note, that unlike in the first example, here the columns "dist" and
# "dist_in_segment" are not identical, as this time profile consists of
# multiple segments. The following figure illustrates the difference:
plt.rcdefaults()
ax: plt.Axes
fig, ax = plt.subplots(1, 1, figsize=(7, 3))
ax.plot(ms_THM_sp["dist"], label="dist")
ax.plot(ms_THM_sp["dist_in_segment"], label="dist_in_segment")
ax.set_xlabel("Point ID / -")
ax.set_ylabel("Distance / m")
ax.legend()
fig.tight_layout()

# %% [markdown]
# The orange line returns to 0 twice. It is because of how the overlap of nodal
# points between segments is handled. A nodal point always belongs to the
# segment it starts: point B is included in segment BC but not AB and point
# C in CD but not in in BC. The following figure shows the profile on the mesh:

# %%
# plt.rcdefaults()
fig, ax = mesh.plot_linesample_contourf(
    [ogs.variables.pressure, ogs.variables.temperature],
    profile_THM,
    resolution=100,
)
# %%
