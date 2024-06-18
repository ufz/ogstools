"""
Stress analysis
===============

.. sectionauthor:: Florian Zill (Helmholtz Centre for Environmental Research GmbH - UFZ)

The following example from the ogs benchmark collection is used for the
stress analysis:

<https://www.opengeosys.org/docs/benchmarks/thermo-mechanics/creepafterexcavation/>


"""

# %%

# sphinx_gallery_start_ignore

# sphinx_gallery_thumbnail_number = -1

# sphinx_gallery_end_ignore


import ogstools as ot
from ogstools import examples

mesh = examples.load_mesh_mechanics_2D()
fig = mesh.plot_contourf(ot.properties.displacement)

# %% [markdown]
# Tensor components
# -----------------
# We can inspect the stress (or strain) tensor components by indexing.

# %%
fig = mesh.plot_contourf(ot.properties.stress["xx"])
fig = mesh.plot_contourf(ot.properties.stress["xy"])

# %% [markdown]
# Principal stresses
# ------------------
# Let's plot the the principal stress components and also overlay the direction
# of the corresponding eigenvector in the plot. Note: the eigenvalues are sorted
# by increasing order, i.e. eigenvalue[0] is the most negative / largest
# compressive principal stress.

# %%
eigvecs = ot.properties.stress.eigenvectors
fig = mesh.plot_contourf(mesh_property=ot.properties.stress.eigenvalues[0])
mesh.plot_quiver(ax=fig.axes[0], mesh_property=eigvecs[0], glyph_type="line")

# %%
fig = mesh.plot_contourf(mesh_property=ot.properties.stress.eigenvalues[1])
mesh.plot_quiver(ax=fig.axes[0], mesh_property=eigvecs[1], glyph_type="line")

# %%
fig = mesh.plot_contourf(mesh_property=ot.properties.stress.eigenvalues[2])
mesh.plot_quiver(ax=fig.axes[0], mesh_property=eigvecs[2], glyph_type="line")

# %% [markdown]
# We can also plot the mean of the principal stress, i.e. the magnitude of the
# hydrostatic component of the stress tensor.
# see: :py:func:`ogstools.propertylib.tensor_math.mean`

# %%
fig = mesh.plot_contourf(ot.properties.stress.mean)

# %% [markdown]
# Von Mises stress
# ----------------
# see: :py:func:`ogstools.propertylib.tensor_math.von_mises`

# %%
fig = mesh.plot_contourf(ot.properties.stress.von_Mises)

# %% [markdown]
# octahedral shear stress
# -----------------------
# see: :py:func:`ogstools.propertylib.tensor_math.octahedral_shear`

# %%
fig = mesh.plot_contourf(ot.properties.stress.octahedral_shear)

# %% [markdown]
# Integrity criteria
# ==================
# Evaluating models regarding their integrity is often dependent on the
# geometry, e.g. for a hypothetical water column proportional to the depth.
# Presets which fall under this category make use of
# :py:mod:`ogstools.propertylib.mesh_dependent`.

# %% [markdown]
# The hypothetical water column used in the integrity criteria would initially
# use existing "pressure" data in the mesh, otherwise it is automatically
# calculated as the following:

# %%
mesh["pressure"] = mesh.p_fluid()
fig = mesh.plot_contourf(ot.properties.pressure)

# %% [markdown]
# But since this assumes that the top of the model is equal to the ground
# surface, the resulting pressure is underestimated. In this case we have to
# correct the depth manually. Then the pressure is calculated correctly:

# %%
mesh["depth"] = mesh.depth(use_coords=True)
fig = mesh.plot_contourf("depth")
mesh["pressure"] = mesh.p_fluid()
fig = mesh.plot_contourf(ot.properties.pressure)

# %% [markdown]
# Dilantancy criterion
# --------------------
# see: :py:func:`ogstools.propertylib.mesh_dependent.dilatancy_critescu`

# %%
fig = mesh.plot_contourf(ot.properties.dilatancy_critescu_tot)
fig = mesh.plot_contourf(ot.properties.dilatancy_critescu_eff)

# %% [markdown]
# Fluid pressure criterion
# ------------------------
# see: :py:func:`ogstools.propertylib.mesh_dependent.fluid_pressure_criterion`

# %%
fig = mesh.plot_contourf(ot.properties.fluid_pressure_crit)
