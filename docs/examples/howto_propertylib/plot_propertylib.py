"""
Property Presets
================

.. sectionauthor:: Florian Zill (Helmholtz Centre for Environmental Research GmbH - UFZ)

:py:mod:`ogstools.propertylib` provides classes (Scalar, Vector, Matrix) which
encapsulate unit handling and data transformation for simplified processing of
mesh data. There are several predefined properties stored under the module
:py:mod:`ogstools.propertylib.properties`.
"""

# %%
from ogstools import examples
from ogstools.meshplotlib import plot
from ogstools.propertylib import Scalar, properties

properties.get_dataframe()

# %% [markdown]
# Scalar, Vector and Matrix inherit from the class Property with its
# :meth:`~ogstools.propertylib.Property.transform` function.
# This function converts the argument from data_unit to output_unit and
# applies a function if specified. In this case we convert from K to °C:

# %%
properties.temperature.transform(273.15)

# %% [markdown]
# You can also create your own properties by creating a Scalar, Vector or Matrix
# property. The following doesn't do any unit conversion.

# %%
custom_temperature = Scalar(
    data_name="temperature", data_unit="K", output_unit="K"
)
custom_temperature.transform(273.15)

# %% [markdown]
# Or use existing presets as a template and replace some parameters:
custom_temperature = properties.temperature.replace(output_unit="°F")
custom_temperature.transform(273.15)

# %% [markdown]
# Components of Vector properties and Matrix properties can be accessed with
# bracket indexing. :class:`~ogstools.propertylib.vector.Vector` properties
# should be of length 2 or 3 corresponding to the dimension.
# :class:`~ogstools.propertylib.matrix.Matrix` properties likewise should be of
# length 4 [xx, yy, zz, xy] or 6 [xx, yy, zz, xy, yz, xz].

# %%
properties.displacement[1].transform([0.01, 0.02, 0.03])

# %%
properties.strain["xx"].transform([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])

# %% [markdown]
# Magnitude of a 2D displacement vector:

# %%
properties.displacement.magnitude.transform([0.03, 0.04])

# %% [markdown]
# We suggest specifying the properties and their transformations once.
# These can be reused in different kind of post processing. When plotting
# with :py:mod:`ogstools.meshplotlib` we can use these presets to simplify the
# task of processing the data (e.g. calculate the von Mises stress):

# %%
fig = plot(examples.load_mesh_mechanics_2D(), properties.stress.von_Mises)

# %% [markdown]
# Have a look at
# :ref:`sphx_glr_auto_examples_howto_meshplotlib` for more examples.
