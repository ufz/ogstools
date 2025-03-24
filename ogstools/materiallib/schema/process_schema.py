# ogstools/materiallib/schema/process_schema.py

PROCESS_SCHEMAS = {
    "HeatConduction": {
        "thermal_conductivity": "medium",
        "density": "medium",
        "specific_heat_capacity": "medium"
    },
    "TH2M": {
        "density": "solid",
        "specific_heat_capacity": "solid",
        "thermal_conductivity": "solid",
        "thermal_expansivity": "solid",
        "porosity": "medium",
        "permeability": "medium",
        "saturation": "medium",
        "bishops_effective_stress": "medium"
    }
}