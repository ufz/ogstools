PROCESS_SCHEMAS = {
    "HeatConduction": {
        "thermal_conductivity": "medium",
        "density": "medium",
        "specific_heat_capacity": "medium",
        "_fluids": {},
    },
    "TH2M": {
        # Solid phase
        "density": "solid",
        "specific_heat_capacity": "solid",
        "thermal_conductivity": "solid",
        "thermal_expansivity": "solid",
        # Medium properties
        "porosity": "medium",
        "permeability": "medium",
        "saturation": "medium",
        "bishops_effective_stress": "medium",
        # Fluid properties (without phase transition)
        "_fluids": {
            "AqueousLiquid": {
                "phase_properties": [
                    "density",
                    "viscosity",
                    "thermal_conductivity",
                    "specific_heat_capacity",
                ]
            },
            "Gas": {
                "phase_properties": [
                    "density",
                    "viscosity",
                    "thermal_conductivity",
                ]
            },
        },
    },
    "TH2M_PT": {
        # Solid phase
        "density": "solid",
        "specific_heat_capacity": "solid",
        "thermal_conductivity": "solid",
        "thermal_expansivity": "solid",
        # Medium properties
        "porosity": "medium",
        "permeability": "medium",
        "saturation": "medium",
        "bishops_effective_stress": "medium",
        # Fluid properties with phase transitions - component properties
        "_fluids": {
            "AqueousLiquid": {
                "component_properties": [
                    "molar_mass",
                    "specific_heat_capacity",
                    "diffusion",
                    "henry_coefficient",
                    "specific_latent_heat",
                ],
                "phase_properties": [
                    "density",
                    "viscosity",
                    "thermal_conductivity",
                    "specific_heat_capacity",
                ],
            },
            "Gas": {
                "component_properties": [
                    "molar_mass",
                    "specific_heat_capacity",
                    "specific_latent_heat",
                    "vapour_pressure",
                    "diffusion",
                ],
                "phase_properties": [
                    "density",
                    "viscosity",
                    "thermal_conductivity",
                ],
            },
        },
    },
}
