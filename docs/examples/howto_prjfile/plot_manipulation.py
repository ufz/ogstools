"""
How to Manipulate Prj-Files
===========================

.. sectionauthor:: Jörg Buchwald (Helmholtz Centre for Environmental Research GmbH - UFZ)

E.g., to iterate over three Young's moduli one can use the replace parameter method.

"""

# %%
from pathlib import Path
from tempfile import mkdtemp

import ogstools as ot
from ogstools.definitions import EXAMPLES_DIR

model_dir = Path(mkdtemp())
youngs_moduli = [1, 2, 3]
filename = EXAMPLES_DIR / "prj/simple_mechanics.prj"
for youngs_modulus in youngs_moduli:
    prj = ot.Project(input_file=filename, output_file=model_dir / filename)
    prj.replace_parameter_value(name="E", value=youngs_modulus)
    prj.replace_text(
        f"out_E={youngs_modulus}", xpath="./time_loop/output/prefix"
    )
    prj.write_input()
    prj.run_model(args=f"-m {EXAMPLES_DIR}/prj/ -o {model_dir}")

# %%
# Instead of the ``replace_parameter`` method, the more general ``replace_text`` method
# can also be used to replace the young modulus in this example:
prj.replace_text(youngs_modulus, xpath="./parameters/parameter[name='E']/value")

# %%
# The Young's modulus in this file can also be accessed through 0'th occurrence of the place addressed by the xpath
prj.replace_text(
    youngs_modulus, xpath="./parameters/parameter/value", occurrence=0
)

# %%
# The density can also be changed:
prj.replace_phase_property_value(
    mediumid=0, phase="Solid", name="density", value="42"
)

# %% [markdown]
# For MPL (Material Property Library) based processes, like TH2M or Thermo-Richards-Mechanics,
# there exist specific functions to set phase and medium properties: e.g.:,
# `model.replace_medium_property_value(mediumid=0, name="porosity", value="0.24")`
