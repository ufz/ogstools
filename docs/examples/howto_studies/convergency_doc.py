"""
Features of studies
=====================================

.. sectionauthor:: Tobias Meisel (Helmholtz Centre for Environmental Research GmbH - UFZ)

``studies`` provides a utility function to compose studies from multiple simulation runs
"""


# %%

import matplotlib.pyplot as plt
import pandas as pd

from ogstools.propertylib.defaults import temperature
from ogstools.studies.convergence import convergence, plot_convergence

# %%
# Run multiple simulations with different mesh discretization, keep everything else constant
# Choice of time-stepping settings and solver tolerance significantly influences convergence
# ogs_dir =

# ToDo Replace below by liquid flow example
# "Tests/Data/Parabolic/HT/SimpleSynthetics/XDMF/CoupledPressureParabolicTemperatureParabolicStaggered.prj"

# The results
# path = "/home/meisel/gitlabrepos/thedi-workflow/results/sim/hostrock-clay_dim-2_invtype-DWR-UOX_depth-600_gtgradient-0.02_refinement-{refinement}/transient/result_result.xdmf"
# timeseries_files = [
#    path.format(refinement=refinement) for refinement in [0, 1, 2]
# ]


# %%
timeseries_files = []
ts = 100
temperature.data_name = "Temperature"
properties = [temperature]

# %%
conv = convergence(timeseries_files, 100, [temperature])
print(conv)

# %%


pd.DataFrame(conv)

# %%
plt.ioff()
fig, axs = plt.subplots(
    dpi=200, figsize=[5, 3], facecolor="white", nrows=len(properties)
)
if not isinstance(axs, list):
    axs = [axs]

plot_convergence(timeseries_files, 100, properties, axs)

# %%
