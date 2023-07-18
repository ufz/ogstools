"""
Features of propertylib
=====================================

.. sectionauthor:: Florian Zill (Helmholtz Centre for Environmental Research GmbH - UFZ)

``propertylib`` provides a common interface for other modules to structure
reading, conversion and output of mesh data.
"""

# %%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ogstools.propertylib as ptl

# %%
# There are some predefined default properties:

tab = pd.DataFrame(ptl.defaults.all_properties).set_index("output_name")
tab["type"] = [type(p).__name__ for p in ptl.defaults.all_properties]
tab.drop(["tag", "func"], axis=1).sort_values(["mask", "data_name"])

# %%
# You can access properties either form the entire collection or from a subset
# which only contains properties available to a specific OGS process.
# Calling a property converts the argument from data_unit to output_unit and
# applies a function if specified.

print(ptl.defaults.temperature(273.15))  # access from the entire collection
print(ptl.M.strain(0.01))  # access from Mechanics collection

# %%
# Available processes are:

print([type(p).__name__ for p in ptl.processes])

# %%
# VectorProperties and MatrixProperties contain other Properties which represent
# the result of an applied function on itself. Components can be accessed with
# brackets. VectorProperties should be of length 2 or 3 corresponding to the
# dimension. MatrixProperties likewise should be of length 4 [xx, yy, zz, xy]
# or 6 [xx, yy, zz, xy, yz, xz].

# %%
# Element 1 (counting from 0) of a 3D displacement vector:

print(ptl.M.displacement[1]([0.01, 0.02, 0.03]))

# %%
# Magnitude of a 2D displacement vector from:

print(ptl.M.displacement.magnitude([0.03, 0.04]))

# %%
# Log of Magnitude of a 2D velocity vector from the Hydraulics collection:
print(ptl.H.velocity.log_magnitude(np.sqrt([50, 50])))

# %%
# Magnitude and trace of a 3D strain matrix:
eps = np.array([1, 3, 9, 1, 2, 2]) * 1e-2
print(ptl.M.strain.magnitude(eps))
print(ptl.M.strain.trace(eps))

# %%
# You can change the attributes of the defaults.
# For example for temperature from the Thermal Collection from the default
# output_unit °C to °F:

temp = np.linspace(273.15, 373.15, 10)
fig, axs = plt.subplots(2)
axs[0].plot(ptl.T.temperature(temp), color="r")
ptl.defaults.temperature.output_unit = "°F"
axs[1].plot(ptl.T.temperature(temp), color="b")
fig.show()
