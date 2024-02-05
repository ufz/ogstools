from pint import UnitRegistry

u_reg: UnitRegistry = UnitRegistry(
    preprocessors=[lambda s: s.replace("%", "percent")]
)
u_reg.default_format = "~.12g"
u_reg.setup_matplotlib(True)
