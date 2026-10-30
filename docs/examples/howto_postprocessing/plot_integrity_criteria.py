"""
Evaluation of mechanical integrity criteria
===========================================

The following example from the ogs benchmark collection is used for the
analysis of integrity criteria.

<https://www.opengeosys.org/stable/docs/benchmarks/thermo-mechanics/creepafterexcavation/>

For evaluation of mechanical integrity criteria we provide Variables, which
apply a transformation function to the stress data ("sigma") in the dataset.
Tension is positive and the pore pressure is positive in compression.

All integrity criteria take the effective stress as input, which is what OGS
writes to "sigma" (for purely mechanical processes effective and total stress
coincide). If the stress in the dataset is total stress, use
``<integrity_variable>.from_total_stress(biot=..., bishop=...)``.
This returns a new Variable, which converts the stress prior to evaluating the
integrity criterion. Bishop's coefficient can be a constant or the name of a
data field (e.g. :math:`\\chi(S_L)` evaluated from a material property). Cell
data is interpolated to the points if needed.

At first, we add a datafield, defining where the integrity criteria are
evaluated, by masking all non-hostrock material ids.
"""

# %%
import ogstools as ot
from ogstools import examples

mesh = examples.load_mesh_mechanics_2D()
mesh[ot.variables.INTEGRITY_MASK] = (mesh["MaterialIDs"] == 1).astype(int)


# %% [markdown]
# Dilatancy criteria
# ==================

# %% [markdown]
# For salt rock
# -------------
# For dilatancy criteria for salt rock, ogstools provides 2 Variables:
# :func:`ogstools.variables.integrity.dilatancy_critescu` and
# :func:`ogstools.variables.integrity.dilatancy_alkan`. They have
# parameters with default values which you can change on a copy like this:

# %%
dil_var = ot.variables.dilatancy_critescu.copy()
dil_var.function.params

# %%
dil_var.function.params["b"] = 0.9
fig = ot.plot.contourf(mesh, dil_var)


# %% [markdown]
# Fluid pressure criterion
# ========================

# %% [markdown]
# For clay rock
# -------------

# %%
fig = ot.plot.contourf(mesh, ot.variables.fluid_pressure_criterion_clay)

# %% [markdown]
# For salt rock
# -------------
# As we usually simulate salt rock with a hydraulically deactivated subdomain,
# there is no distinction between its effective and total stress and the
# pressure field in this case is zero. It is common, to use a hypothetical water
# column to evaluate, if fluid pathways could open or not. For this purpose, the
# Variable we provide expects to have a field named 'water_pressure' containing
# the hypothetical pressure data.

# %%
mesh["water_pressure"] = -1000 * 9.81 * mesh.points[:, 1]  # rho * g * h
# if the vertical coordinate does not correspond with the depth below the ground
# level, use `ot.mesh.depth(mesh, top_mesh)` instead.
fig = ot.plot.contourf(mesh, ot.variables.fluid_pressure_criterion_salt)
