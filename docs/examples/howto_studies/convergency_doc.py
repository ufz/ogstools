# %%

import matplotlib.pyplot as plt
import pandas as pd

from ogstools.propertylib import defaults
from ogstools.study.convergence import convergence, plot_convergence

# %%
# Run multiple simulations with different mesh discretization, keep everything else constant
# The coarse the time_step the more visible
# Adapt solver tolerance when needed
# ogs_dir =

# "Tests/Data/Parabolic/HT/SimpleSynthetics/XDMF/CoupledPressureParabolicTemperatureParabolicStaggered.prj"

# The results
# path = "/home/meisel/gitlabrepos/thedi-workflow/results/sim/hostrock-clay_dim-2_invtype-DWR-UOX_depth-600_gtgradient-0.02_refinement-{refinement}/transient/result_result.xdmf"
# timeseries_files = [
#    path.format(refinement=refinement) for refinement in [0, 1, 2]
# ]


# %%
timeseries_files = []
ts = 100
defaults.temperature.data_name = "Temperature"
properties = [defaults.temperature]

# %%
conv = convergence(timeseries_files, 100, [defaults.temperature])
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
