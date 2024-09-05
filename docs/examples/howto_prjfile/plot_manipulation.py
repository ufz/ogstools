"""
How to Manipulate Prj-Files
===========================

.. sectionauthor:: Jörg Buchwald (Helmholtz Centre for Environmental Research GmbH - UFZ)

E.g., to iterate over three Young's moduli one can use the replace parameter method.

"""

# %%
# 1. Initialize the ogs6py object:
from ogstools.definitions import EXAMPLES_DIR
from ogstools.ogs6py import ogs

Es = [1, 2, 3]
filename = EXAMPLES_DIR / "prj/simple_mechanics.prj"
for E in Es:
    model = ogs.OGS(INPUT_FILE=filename, PROJECT_FILE=filename)
    model.replace_parameter_value(name="E", value=E)
    model.replace_text("out_E=" + str(E), xpath="./time_loop/output/prefix")
    model.write_input()
    model.run_model()

# %%
# 2. Instead of the `replace_parameter` method, the more general `replace_text` method
# can also be used to replace the young modulus in this example:
model.replace_text(E, xpath="./parameters/parameter[name='E']/value")

# %%
# 3. The Young's modulus in this file can also be accessed through 0'th occurrence of the place addressed by the xpath
model.replace_text(E, xpath="./parameters/parameter/value", occurrence=0)

# %%
# 4. The density can also be changed:
model.replace_phase_property_value(
    mediumid=0, phase="Solid", name="density", value="42"
)

# %% [markdown]
# 5. For MPL (Material Property Library) based processes, like TH2M or Thermo-Richards-Mechanics,
# there exist specific functions to set phase and medium properties: e.g.:,
# `model.replace_medium_property_value(mediumid=0, name="porosity", value="0.24")`
