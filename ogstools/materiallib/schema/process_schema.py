# SPDX-FileCopyrightText: Copyright (c) OpenGeoSys Community (opengeosys.org)
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

_T_PROPS = ["density", "specific_heat_capacity"]
_T_SOLID = _T_PROPS + ["thermal_conductivity"]
_BHE_LIQUID = _T_PROPS + ["phase_velocity"]
_H_LIQUID = ["density", "viscosity"]

_TH_LIQUID = _T_SOLID + ["viscosity"]
_TH_SOLID = _T_SOLID + ["storage"]
_TM_SOLID = _T_SOLID + ["thermal_expansivity"]
_THM_SOLID = _TH_SOLID + ["thermal_expansivity"]

_MED_PROPS = ["permeability", "porosity"]
_H_MED_PROPS = _MED_PROPS + ["storage", "reference_temperature"]
_HM_MED_PROPS = _MED_PROPS + ["biot_coefficient", "reference_temperature"]
_T_DISP_PROPS = [
    "thermal_conductivity",
    "thermal_longitudinal_dispersivity",
    "thermal_transversal_dispersivity",
]
_TH_MED_PROPS = _MED_PROPS + _T_DISP_PROPS
_THM_MED_PROPS = _TH_MED_PROPS + ["biot_coefficient"]
_BHE_MED_PROPS = _T_DISP_PROPS + ["porosity"]


def _to_schema_dict(phases: dict, med_props: list) -> dict:
    phase_props = [{"type": k, "properties": v} for k, v in phases.items()]
    return {"phases": phase_props, "properties": med_props}


PROCESS_SCHEMAS: dict[str, dict[str, Any]] = {
    "HEAT_CONDUCTION": _to_schema_dict({}, _T_SOLID),
    "LIQUID_FLOW": _to_schema_dict({"AqueousLiquid": _H_LIQUID}, _H_MED_PROPS),
    "SMALL_DEFORMATION": _to_schema_dict({"Solid": ["density"]}, []),
    "TM": _to_schema_dict({"Solid": _TM_SOLID}, []),
    "HM": _to_schema_dict(
        {"AqueousLiquid": _H_LIQUID, "Solid": ["density"]}, _HM_MED_PROPS
    ),
    "TH": _to_schema_dict(
        {"AqueousLiquid": _TH_LIQUID, "Solid": _TH_SOLID}, _TH_MED_PROPS
    ),
    "THM": _to_schema_dict(
        {"AqueousLiquid": _TH_LIQUID, "Solid": _THM_SOLID}, _THM_MED_PROPS
    ),
    "HEAT_TRANSPORT_BHE": _to_schema_dict(
        {"AqueousLiquid": _BHE_LIQUID, "Solid": _T_PROPS}, _BHE_MED_PROPS
    ),
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
                "properties": _TH_LIQUID,
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
            {"type": "Solid", "properties": _TM_SOLID},
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
# aliases for practicality
PROCESS_SCHEMAS["T"] = PROCESS_SCHEMAS["HEAT_CONDUCTION"]
PROCESS_SCHEMAS["H"] = PROCESS_SCHEMAS["LIQUID_FLOW"]
PROCESS_SCHEMAS["M"] = PROCESS_SCHEMAS["SMALL_DEFORMATION"]


# PROCESS_SCHEMAS = {
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
# }
