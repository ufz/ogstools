"""
Feflowlib: How to get started with the FEFLOW converter.
========================================================

.. sectionauthor:: Julian Heinze (Helmholtz Centre for Environmental Research GmbH - UFZ)

This example shows how a FEFLOW model can be converted and simulated in OGS.
"""

# %%
# 0. Necessary imports.
import tempfile
from pathlib import Path

from ogstools import FeflowModel
from ogstools.examples import (
    feflow_model_2D_CT_t_560,
    feflow_model_2D_HT,
    feflow_model_box_Robin,
)

# %%
# 1. Convert the models.
temp_dir = Path(tempfile.mkdtemp("feflow_test_simulation"))
feflow_model_H = FeflowModel(
    feflow_model_box_Robin, temp_dir / "3D_H_model.vtu"
)
feflow_model_HC = FeflowModel(
    feflow_model_2D_CT_t_560, temp_dir / "2D_HC_model.vtu"
)
feflow_model_HT = FeflowModel(feflow_model_2D_HT, temp_dir / "2D_HT_model.vtu")
# %%
# 2. Define simulation times.
# Simulate the steady state diffusion process in OGS for the H-model.
feflow_model_H.setup_prj(steady=True)
feflow_model_HC.setup_prj(
    end_time=int(4.8384e07),
    time_stepping=list(
        zip([10] * 8, [8.64 * 10**i for i in range(8)], strict=False)
    ),
)
feflow_model_HT.setup_prj(end_time=1e11, time_stepping=[(1, 1e10)])
# %%
# 3. Run the simulations.
feflow_model_H.run()
feflow_model_HC.run()
feflow_model_HT.run()
