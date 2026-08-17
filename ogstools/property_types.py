# SPDX-FileCopyrightText: Copyright (c) OpenGeoSys Community (opengeosys.org)
# SPDX-License-Identifier: BSD-3-Clause

"""Shared registry of OGS material property type specifications."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PropertyTypeSpec:
    """Describe the expected parameters and metadata of one property type."""

    parameters: tuple[str, ...]
    metadata_keys: tuple[str, ...] = ("unit", "source")


PROPERTY_TYPES: dict[str, PropertyTypeSpec] = {
    "AverageMolarMass": PropertyTypeSpec(parameters=()),
    "BishopsSaturationCutoff": PropertyTypeSpec(parameters=("cutoff_value",)),
    "BishopsPowerLaw": PropertyTypeSpec(parameters=("exponent",)),
    "CapillaryPressureRegularizedVanGenuchten": PropertyTypeSpec(
        parameters=(
            "exponent",
            "p_b",
            "residual_gas_saturation",
            "residual_liquid_saturation",
        )
    ),
    "CapillaryPressureVanGenuchten": PropertyTypeSpec(
        parameters=(
            "exponent",
            "maximum_capillary_pressure",
            "p_b",
            "residual_gas_saturation",
            "residual_liquid_saturation",
        )
    ),
    "ClausiusClapeyron": PropertyTypeSpec(
        parameters=(
            "critical_pressure",
            "critical_temperature",
            "reference_pressure",
            "reference_temperature",
            "triple_pressure",
            "triple_temperature",
        )
    ),
    "Constant": PropertyTypeSpec(parameters=("value",)),
    "Curve": PropertyTypeSpec(parameters=("curve", "independent_variable")),
    "DupuitPermeability": PropertyTypeSpec(parameters=("parameter_name",)),
    "EffectiveThermalConductivityPorosityMixing": PropertyTypeSpec(
        parameters=()
    ),
    "EmbeddedFracturePermeability": PropertyTypeSpec(
        parameters=(
            "intrinsic_permeability",
            "initial_aperture",
            "mean_frac_distance",
            "threshold_strain",
            "fracture_normal",
            "fracture_rotation_xy",
            "fracture_rotation_yz",
        )
    ),
    "Exponential": PropertyTypeSpec(parameters=("offset", "reference_value")),
    "Function": PropertyTypeSpec(parameters=("value",)),
    "GasPressureDependentPermeability": PropertyTypeSpec(
        parameters=(
            "initial_permeability",
            "a1",
            "a2",
            "pressure_threshold",
            "minimum_permeability",
            "maximum_permeability",
        )
    ),
    "IdealGasLaw": PropertyTypeSpec(parameters=()),
    "IdealGasLawBinaryMixture": PropertyTypeSpec(parameters=()),
    "KozenyCarmanModel": PropertyTypeSpec(
        parameters=(
            "initial_permeability",
            "initial_porosity",
        )
    ),
    "Linear": PropertyTypeSpec(parameters=("reference_value",)),
    "LinearSaturationSwellingStress": PropertyTypeSpec(
        parameters=(
            "coefficient",
            "reference_saturation",
        )
    ),
    "LinearWaterVapourLatentHeat": PropertyTypeSpec(parameters=()),
    "OrthotropicEmbeddedFracturePermeability": PropertyTypeSpec(
        parameters=(
            "intrinsic_permeability",
            "mean_frac_distances",
            "threshold_strains",
            "fracture_normals",
            "fracture_rotation_xy",
            "fracture_rotation_yz",
            "jacobian_factor",
        )
    ),
    "Parameter": PropertyTypeSpec(parameters=("parameter_name",)),
    "PermeabilityMohrCoulombFailureIndexModel": PropertyTypeSpec(
        parameters=(
            "cohesion",
            "fitting_factor",
            "friction_angle",
            "initial_permeability",
            "maximum_permeability",
            "reference_permeability",
            "tensile_strength_parameter",
        )
    ),
    "PorosityFromMassBalance": PropertyTypeSpec(
        parameters=(
            "initial_porosity",
            "maximal_porosity",
            "minimal_porosity",
        )
    ),
    "RelPermBrooksCorey": PropertyTypeSpec(
        parameters=(
            "lambda",
            "min_relative_permeability",
            "residual_gas_saturation",
            "residual_liquid_saturation",
        )
    ),
    "RelPermBrooksCoreyNonwettingPhase": PropertyTypeSpec(
        parameters=(
            "lambda",
            "min_relative_permeability",
            "residual_gas_saturation",
            "residual_liquid_saturation",
        )
    ),
    "RelPermLiakopoulos": PropertyTypeSpec(parameters=()),
    "RelativePermeabilityNonWettingVanGenuchten": PropertyTypeSpec(
        parameters=(
            "exponent",
            "minimum_relative_permeability",
            "residual_gas_saturation",
            "residual_liquid_saturation",
        )
    ),
    "RelativePermeabilityUdell": PropertyTypeSpec(
        parameters=(
            "min_relative_permeability",
            "residual_gas_saturation",
            "residual_liquid_saturation",
        )
    ),
    "RelativePermeabilityUdellNonwettingPhase": PropertyTypeSpec(
        parameters=(
            "min_relative_permeability",
            "residual_gas_saturation",
            "residual_liquid_saturation",
        )
    ),
    "RelativePermeabilityVanGenuchten": PropertyTypeSpec(
        parameters=(
            "exponent",
            "minimum_relative_permeability_liquid",
            "residual_gas_saturation",
            "residual_liquid_saturation",
        )
    ),
    "RelativePermeabilityNonWettingPhaseVanGenuchtenMualem": PropertyTypeSpec(
        parameters=(
            "exponent",
            "min_relative_permeability",
            "residual_gas_saturation",
            "residual_liquid_saturation",
        )
    ),
    "PermeabilityOrthotropicPowerLaw": PropertyTypeSpec(
        parameters=("exponents", "intrinsic_permeabilities")
    ),
    "SaturationBrooksCorey": PropertyTypeSpec(
        parameters=(
            "entry_pressure",
            "lambda",
            "residual_gas_saturation",
            "residual_liquid_saturation",
        )
    ),
    "SaturationDependentSwelling": PropertyTypeSpec(
        parameters=(
            "exponents",
            "lower_saturation_limit",
            "swelling_pressures",
            "upper_saturation_limit",
        )
    ),
    "SaturationDependentThermalConductivity": PropertyTypeSpec(
        parameters=("dry", "wet")
    ),
    "SaturationExponential": PropertyTypeSpec(
        parameters=(
            "exponent",
            "maximum_capillary_pressure",
            "residual_gas_saturation",
            "residual_liquid_saturation",
        )
    ),
    "SaturationLiakopoulos": PropertyTypeSpec(parameters=()),
    "SaturationVanGenuchten": PropertyTypeSpec(
        parameters=(
            "exponent",
            "p_b",
            "residual_gas_saturation",
            "residual_liquid_saturation",
        )
    ),
    "SaturationWeightedThermalConductivity": PropertyTypeSpec(
        parameters=(
            "mean_type",
            "dry_thermal_conductivity",
            "wet_thermal_conductivity",
        )
    ),
    "SoilThermalConductivitySomerton": PropertyTypeSpec(
        parameters=(
            "dry_thermal_conductivity",
            "wet_thermal_conductivity",
        )
    ),
    "StrainDependentPermeability": PropertyTypeSpec(
        parameters=(
            "initial_permeability",
            "b1",
            "b2",
            "b3",
            "minimum_permeability",
            "maximum_permeability",
        )
    ),
    "TemperatureDependentDiffusion": PropertyTypeSpec(
        parameters=(
            "activation_energy",
            "reference_diffusion",
            "reference_temperature",
        )
    ),
    "TransportPorosityFromMassBalance": PropertyTypeSpec(
        parameters=(
            "initial_porosity",
            "maximal_porosity",
            "minimal_porosity",
        )
    ),
    "VapourDiffusionFEBEX": PropertyTypeSpec(parameters=()),
    "VapourDiffusionPMQ": PropertyTypeSpec(parameters=()),
    "VermaPruessModel": PropertyTypeSpec(
        parameters=(
            "critical_porosity",
            "exponent",
            "initial_permeability",
            "initial_porosity",
        )
    ),
    "WaterVapourDensity": PropertyTypeSpec(parameters=()),
    "WaterDensityIAPWSIF97Region1": PropertyTypeSpec(parameters=()),
    "WaterVapourLatentHeatWithCriticalTemperature": PropertyTypeSpec(
        parameters=()
    ),
    "WaterViscosityIAPWS": PropertyTypeSpec(parameters=()),
}
