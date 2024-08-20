"""
Variable presets and data transformation
========================================

.. sectionauthor:: Florian Zill (Helmholtz Centre for Environmental Research GmbH - UFZ)

:py:mod:`ogstools.variables` provides classes (Scalar, Vector, Matrix) which
encapsulate unit handling and data transformation for simplified processing of
mesh data. There are also several predefined variables.
"""

# %%
import ogstools as ot
from ogstools import examples

ot.variables.get_dataframe()

# %% [markdown]
# Scalar, Vector and Matrix inherit from the class Variable with its
# :meth:`~ogstools.variables.variable.Variable.transform` function.
# This function converts the argument from data_unit to output_unit and
# applies a function if specified. In this case we convert from K to °C:

# %%
ot.variables.temperature.transform(273.15, strip_unit=False)

# %% [markdown]
# You can also create your own variables by creating a Scalar, Vector or Matrix
# variable. The following doesn't do any unit conversion.

# %%
custom_temperature = ot.variables.Scalar(
    data_name="temperature", data_unit="K", output_unit="K"
)
custom_temperature.transform(273.15, strip_unit=False)

# %% [markdown]
# Or use existing presets as a template and replace some parameters:
custom_temperature = ot.variables.temperature.replace(output_unit="°F")
custom_temperature.transform(273.15, strip_unit=False)

# %% [markdown]
# Components of Vector variables and Matrix variables can be accessed with
# bracket indexing. :class:`~ogstools.variables.vector.Vector` variables
# should be of length 2 or 3 corresponding to the dimension.
# :class:`~ogstools.variables.matrix.Matrix` variables likewise should be of
# length 4 [xx, yy, zz, xy] or 6 [xx, yy, zz, xy, yz, xz].

# %%
ot.variables.displacement[1].transform([0.01, 0.02, 0.03], strip_unit=False)

# %%
ot.variables.strain["xx"].transform(
    [0.01, 0.02, 0.03, 0.04, 0.05, 0.06], strip_unit=False
)

# %% [markdown]
# Magnitude of a 2D displacement vector:

# %%
ot.variables.displacement.magnitude.transform([0.03, 0.04], strip_unit=False)

# %% [markdown]
# We suggest specifying the variables and their transformations once.
# These can be reused in different kind of post processing. When plotting
# with :py:mod:`ogstools.plot` we can use these presets to simplify the
# task of processing the data (e.g. calculate the von Mises stress):

# %%
fig = ot.plot.contourf(
    examples.load_mesh_mechanics_2D(), ot.variables.stress.von_Mises
)

# %% [markdown]
# Have a look at
# :ref:`sphx_glr_auto_examples_howto_plot` for more examples.
