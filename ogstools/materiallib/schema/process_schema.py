# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

from typing import Any

PROCESS_SCHEMAS: dict[str, dict[str, Any]] = {
    "SMALL_DEFORMATION": {
        "phases": [{"type": "Solid", "properties": ["density"]}],
        "properties": [],
    },
    "HEAT_CONDUCTION": {
        "phases": [],
        "properties": [
            "thermal_conductivity",
            "density",
            "specific_heat_capacity",
        ],
    },
    "TH2M_PT": {
        "phases": [
            {
                "type": "AqueousLiquid",
                "components": {
                    "Solute": [
                        "specific_heat_capacity",
                        "henry_coefficient",
                        "diffusion",
                        "specific_latent_heat",
                    ],
                    "Solvent": ["specific_heat_capacity"],
                },
                "properties": [
                    "thermal_conductivity",
                    "specific_heat_capacity",
                    "density",
                    "viscosity",
                ],
            },
            {
                "type": "Gas",
                "components": {
                    "Carrier": ["molar_mass", "specific_heat_capacity"],
                    "Vapour": [
                        "molar_mass",
                        "specific_heat_capacity",
                        "specific_latent_heat",
                        "vapour_pressure",
                        "diffusion",
                    ],
                },
                "properties": ["thermal_conductivity", "density", "viscosity"],
            },
            {
                "type": "Solid",
                "properties": [
                    "density",
                    "thermal_conductivity",
                    "specific_heat_capacity",
                    "thermal_expansivity",
                ],
            },
        ],
        "properties": [
            "permeability",
            "biot_coefficient",
            "saturation",
            "relative_permeability_nonwetting_phase",
            "relative_permeability",
            "porosity",
            "tortuosity",
            "thermal_conductivity",
            "bishops_effective_stress",
        ],
    },
}


# PROCESS_SCHEMAS = {
#     "HeatConduction": {
#         "thermal_conductivity": "medium",
#         "density": "medium",
#         "specific_heat_capacity": "medium",
#         "_fluids": {},
#     },
#     "TH2M": {
#         # Solid phase
#         "density": "solid",
#         "specific_heat_capacity": "solid",
#         "thermal_conductivity": "solid",
#         "thermal_expansivity": "solid",
#         # Medium properties
#         "porosity": "medium",
#         "permeability": "medium",
#         "saturation": "medium",
#         "bishops_effective_stress": "medium",
#         # Fluid properties (without phase transition)
#         "_fluids": {
#             "AqueousLiquid": {
#                 "phase_properties": [
#                     "density",
#                     "viscosity",
#                     "thermal_conductivity",
#                     "specific_heat_capacity",
#                 ]
#             },
#             "Gas": {
#                 "phase_properties": [
#                     "density",
#                     "viscosity",
#                     "thermal_conductivity",
#                 ]
#             },
#         },
#     },
#     "TH2M_PT": {
#         # Solid phase
#         "density": "solid",
#         "specific_heat_capacity": "solid",
#         "thermal_conductivity": "solid",
#         "thermal_expansivity": "solid",
#         # Medium properties
#         "porosity": "medium",
#         "permeability": "medium",
#         "saturation": "medium",
#         "bishops_effective_stress": "medium",
#         # Fluid properties with phase transitions - component properties
#         "_fluids": {
#             "AqueousLiquid": {
#                 "component_properties": [
#                     "molar_mass",
#                     "specific_heat_capacity",
#                     "diffusion",
#                     "henry_coefficient",
#                     "specific_latent_heat",
#                 ],
#                 "phase_properties": [
#                     "density",
#                     "viscosity",
#                     "thermal_conductivity",
#                     "specific_heat_capacity",
#                 ],
#             },
#             "Gas": {
#                 "component_properties": [
#                     "molar_mass",
#                     "specific_heat_capacity",
#                     "specific_latent_heat",
#                     "vapour_pressure",
#                     "diffusion",
#                 ],
#                 "phase_properties": [
#                     "density",
#                     "viscosity",
#                     "thermal_conductivity",
#                 ],
#             },
#         },
#     },
# }
