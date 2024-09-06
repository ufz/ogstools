import platform
import shutil
import sys
import tempfile

# this needs to be replaced with regexes from specific ogs version
from collections import defaultdict
from pathlib import Path
from typing import NoReturn

import pytest
from lxml import etree as ET

from ogstools.examples import (
    prj_beier_sandbox,
    prj_beier_sandbox_ref,
    prj_deactivate_replace,
    prj_heat_transport,
    prj_include_solid,
    prj_include_solid_ref,
    prj_pid_timestepping,
    prj_pid_timestepping_ref,
    prj_solid_inc_ref,
    prj_square_1e4_robin,
    prj_square_1e4_robin_ref,
    prj_staggered,
    prj_staggered_ref,
    prj_time_dep_het_param,
    prj_time_dep_het_param_ref,
    prj_trm_from_scratch,
    prj_tunnel_trm,
    prj_tunnel_trm_withincludes,
)
from ogstools.ogs6py import Project


def log_types(records):
    d = defaultdict(list)
    for record in records:
        d[type(record)].append(record)
    return d


mapping = dict.fromkeys(range(32))


class TestiOGS:
    def test_buildfromscratch(self) -> NoReturn:
        model = Project(
            output_file="tunnel_ogs6py.prj", MKL=True, OMP_NUM_THREADS=4
        )
        model.mesh.add_mesh(
            filename="Decovalex-0-simplified-plain-with-p0-plain.vtu"
        )
        model.mesh.add_mesh(
            filename="Decovalex-0-Boundary-Top-mapped-plain.vtu"
        )
        model.mesh.add_mesh(
            filename="Decovalex-0-Boundary-Left-mapped-plain.vtu"
        )
        model.mesh.add_mesh(
            filename="Decovalex-0-Boundary-Bottom-mapped-plain.vtu"
        )
        model.mesh.add_mesh(
            filename="Decovalex-0-Boundary-Right-mapped-plain.vtu"
        )
        model.mesh.add_mesh(
            filename="Decovalex-0-Boundary-Heater-mapped-plain.vtu"
        )
        model.processes.set_process(
            name="Decovalex-0",
            type="THERMO_RICHARDS_MECHANICS",
            integration_order="2",
            specific_body_force="0 0",
            initial_stress="Initial_stress",
            mass_lumping="true",
        )
        model.processes.set_constitutive_relation(
            id="0",
            type="LinearElasticOrthotropic",
            youngs_moduli="YoungsModuliClay",
            shear_moduli="ShearModuliClay",
            poissons_ratios="PoissonsRatiosClay",
        )
        model.processes.set_constitutive_relation(
            id="0",
            type="LinearElasticIsotropic",
            youngs_modulus="YoungsModulusBent",
            poissons_ratio="PoissonsRatioBent",
        )
        model.processes.set_constitutive_relation(
            id="0",
            type="LinearElasticIsotropic",
            youngs_modulus="YoungsModulusBlock",
            poissons_ratio="PoissonsRatioBlock",
        )
        model.processes.add_process_variable(
            process_variable="displacement",
            process_variable_name="displacement",
        )
        model.processes.add_process_variable(
            process_variable="pressure", process_variable_name="pressure"
        )
        model.processes.add_process_variable(
            process_variable="temperature", process_variable_name="temperature"
        )
        model.processes.add_secondary_variable(
            internal_name="saturation", output_name="saturation"
        )
        model.processes.add_secondary_variable(
            internal_name="sigma", output_name="sigma"
        )
        model.processes.add_secondary_variable(
            internal_name="epsilon", output_name="epsilon"
        )
        model.processes.add_secondary_variable(
            internal_name="velocity", output_name="velocity"
        )
        model.processes.add_secondary_variable(
            internal_name="liquid_density", output_name="liquid_density"
        )
        for i in range(3):
            mediumid = str(i)
            model.media.add_property(
                medium_id=mediumid,
                phase_type="Gas",
                name="density",
                type="WaterVapourDensity",
            )
            model.media.add_property(
                medium_id=mediumid,
                phase_type="Gas",
                name="diffusion",
                type="VapourDiffusionFEBEX",
            )
            model.media.add_property(
                medium_id=mediumid,
                phase_type="Gas",
                name="specific_latent_heat",
                type="LinearWaterVapourLatentHeat",
            )
            model.media.add_property(
                medium_id=mediumid,
                phase_type="Gas",
                name="thermal_diffusion_enhancement_factor",
                type="Constant",
                value="1.0",
            )
            model.media.add_property(
                medium_id=mediumid,
                phase_type="Gas",
                name="specific_heat_capacity",
                type="Constant",
                value="0.0",
            )
            model.media.add_property(
                medium_id=mediumid,
                phase_type="AqueousLiquid",
                name="specific_heat_capacity",
                type="Constant",
                value="4181.3",
            )
            model.media.add_property(
                medium_id=mediumid,
                phase_type="AqueousLiquid",
                name="density",
                type="Linear",
                reference_value="1000.0",
                independent_variables={
                    "temperature": {
                        "reference_condition": "273.15",
                        "slope": "-4e-4",
                    },
                    "liquid_phase_pressure": {
                        "reference_condition": "1e5",
                        "slope": "4.6511627906976743356e-10",
                    },
                },
            )
            model.media.add_property(
                medium_id=mediumid,
                phase_type="AqueousLiquid",
                name="viscosity",
                type="Curve",
                curve="ViscosityWater",
                independent_variable="temperature",
            )
        model.media.add_property(
            medium_id="0",
            phase_type="Solid",
            name="specific_heat_capacity",
            type="Constant",
            value="995",
        )
        model.media.add_property(
            medium_id="0",
            phase_type="Solid",
            name="thermal_expansivity",
            type="Constant",
            value="1.5e-5",
        )
        model.media.add_property(
            medium_id="0",
            phase_type="Solid",
            name="density",
            type="Constant",
            value="2689.65517241379",
        )
        for i in range(3):
            mediumid = str(i)
            model.media.add_property(
                medium_id=mediumid,
                name="tortuosity",
                type="Constant",
                value="0.8",
            )
        model.media.add_property(
            medium_id="0", name="porosity", type="Constant", value="0.13"
        )
        for i in range(3):
            mediumid = str(i)
            model.media.add_property(
                medium_id=mediumid,
                name="biot_coefficient",
                type="Constant",
                value="1",
            )
        model.media.add_property(
            medium_id="0",
            name="permeability",
            type="Parameter",
            parameter_name="IntrinsicPermClay",
        )
        model.media.add_property(
            medium_id="0",
            name="relative_permeability",
            type="RelativePermeabilityVanGenuchten",
            residual_liquid_saturation="0",
            residual_gas_saturation="0",
            exponent=0.6,
            minimum_relative_permeability_liquid=1e-6,
        )
        model.media.add_property(
            medium_id="0",
            name="saturation",
            type="SaturationVanGenuchten",
            residual_liquid_saturation="0",
            residual_gas_saturation="0",
            exponent=0.6,
            p_b=20000000,
        )
        for i in range(3):
            mediumid = str(i)
            model.media.add_property(
                medium_id=mediumid,
                name="bishops_effective_stress",
                type="BishopsSaturationCutoff",
                cutoff_value="1",
            )
        model.media.add_property(
            medium_id="0",
            name="thermal_conductivity",
            type="Parameter",
            parameter_name="ThermalConductivityClay",
        )
        model.media.add_property(
            medium_id="1",
            phase_type="Solid",
            name="specific_heat_capacity",
            type="Constant",
            value="800.0",
        )
        model.media.add_property(
            medium_id="1",
            phase_type="Solid",
            name="thermal_expansivity",
            type="Constant",
            value="3e-6",
        )
        model.media.add_property(
            medium_id="1",
            phase_type="Solid",
            name="density",
            type="Constant",
            value="2242.15246636771",
        )
        model.media.add_property(
            medium_id="1", name="porosity", type="Constant", value="0.331"
        )
        model.media.add_property(
            medium_id="1",
            name="permeability",
            type="Constant",
            value="3.5e-20",
        )
        model.media.add_property(
            medium_id="1",
            name="relative_permeability",
            type="RelativePermeabilityVanGenuchten",
            residual_liquid_saturation="0",
            residual_gas_saturation="0",
            exponent=0.5,
            minimum_relative_permeability_liquid=1e-6,
        )
        model.media.add_property(
            medium_id="1",
            name="saturation",
            type="SaturationVanGenuchten",
            residual_liquid_saturation="0",
            residual_gas_saturation="0",
            exponent=0.5,
            p_b=28600000,
        )
        model.media.add_property(
            medium_id="1",
            name="thermal_conductivity",
            type="Curve",
            curve="ThermalConductivityBent",
            independent_variable="liquid_saturation",
        )

        model.media.add_property(
            medium_id=2,
            phase_type="Solid",
            name="specific_heat_capacity",
            type="Constant",
            value="800",
        )
        model.media.add_property(
            medium_id=2,
            phase_type="Solid",
            name="thermal_expansivity",
            type="Constant",
            value="3e-6",
        )
        model.media.add_property(
            medium_id=2,
            phase_type="Solid",
            name="density",
            type="Constant",
            value="2526.15844544096",
        )
        model.media.add_property(
            medium_id="2", name="porosity", type="Constant", value="0.331"
        )
        model.media.add_property(
            medium_id=2,
            name="permeability",
            type="Constant",
            value="1e-22",
        )
        model.media.add_property(
            medium_id="2",
            name="relative_permeability",
            type="RelativePermeabilityVanGenuchten",
            residual_liquid_saturation="0",
            residual_gas_saturation="0",
            exponent=0.4011976,
            minimum_relative_permeability_liquid=1e-6,
        )
        model.media.add_property(
            medium_id="2",
            name="saturation",
            type="SaturationVanGenuchten",
            residual_liquid_saturation="0",
            residual_gas_saturation="0",
            exponent=0.4011976,
            p_b=30000000,
        )
        model.media.add_property(
            medium_id="2",
            name="thermal_conductivity",
            type="Curve",
            curve="ThermalConductivityBlock",
            independent_variable="liquid_saturation",
        )

        model.time_loop.add_process(
            process="Decovalex-0",
            nonlinear_solver_name="basic_newton",
            convergence_type="PerComponentDeltaX",
            norm_type="NORM2",
            reltols="1e-10 1e-10 1e-6 1e-6",
            time_discretization="BackwardEuler",
        )
        model.time_loop.set_stepping(
            process="Decovalex-0",
            type="FixedTimeStepping",
            t_initial=0,
            t_end=864000,
            delta_t=86400,
            repeat=1826,
        )
        model.time_loop.add_time_stepping_pair(
            process="Decovalex-0", repeat=1, delta_t=21600
        )

        model.time_loop.add_output(
            type="VTK",
            prefix="Decovalex-0",
            suffix="_ts_{:timestep}_t_{:time}",
            repeat="1000",
            each_steps="10",
            variables=[
                "displacement",
                "pressure",
                "temperature",
                "sigma",
                "epsilon",
                "velocity",
                "saturation",
                "liquid_density",
            ],
            fixed_output_times=[
                0,
                17280000,
                34560000,
                69120000,
                103680000,
                138240000,
                157788000,
            ],
        )
        model.local_coordinate_system.add_basis_vec(
            basis_vector_0=None, basis_vector_1="b1"
        )
        model.parameters.add_parameter(
            name="b1",
            type="Constant",
            values="-0.55919290347074683016 0.829037572555041692",
        )
        model.parameters.add_parameter(
            name="ThermalConductivityClay",
            type="Constant",
            values="2.4 0 0 1.3",
            use_local_coordinate_system="true",
        )
        model.parameters.add_parameter(
            name="IntrinsicPermClay",
            type="Constant",
            values="5e-20 0 0 1e-20",
            use_local_coordinate_system="true",
        )
        model.parameters.add_parameter(
            name="YoungsModuliClay",
            type="Constant",
            values="8e9 4e9 4e9",
        )
        model.parameters.add_parameter(
            name="YoungsModuliBent",
            type="Constant",
            values="18e6",
        )
        model.parameters.add_parameter(
            name="YoungsModulusBlock",
            type="Constant",
            values="24e6",
        )
        model.parameters.add_parameter(
            name="ShearModuliClay",
            type="Constant",
            values="3.5e9 3.5e9 3.5e9",
        )
        model.parameters.add_parameter(
            name="PoissonsRatiosClay",
            type="Constant",
            values="0.35 0.25 0.25",
        )
        model.parameters.add_parameter(
            name="PoissonsRatioBent",
            type="Constant",
            values="0.35",
        )
        model.parameters.add_parameter(
            name="PoissonsRatioBlock",
            type="Constant",
            values="0.2",
        )
        model.parameters.add_parameter(
            name="Initial_stress",
            type="Constant",
            mesh="Decovalex-0-simplified-plain-with-p0-plain",
            values="-6.5e6 -2.5e6 -4.5e6 0",
        )
        model.parameters.add_parameter(
            name="displacement_ic", type="Constant", values="0 0"
        )

        model.parameters.add_parameter(
            name="pressure_ic", type="MeshNode", field_name="p0"
        )
        model.parameters.add_parameter(
            name="temperature_ic", type="Constant", value="288.15"
        )
        model.parameters.add_parameter(
            name="dirichlet", type="Constant", value="0"
        )
        model.parameters.add_parameter(
            name="heater", type="Constant", value="88.9686017167718"
        )
        model.curves.add_curve(
            name="ThermalConductivityBent",
            coords=[
                0.00,
                0.05,
                0.10,
                0.15,
                0.20,
                0.25,
                0.30,
                0.35,
                0.40,
                0.45,
                0.50,
                0.55,
                0.60,
                0.65,
                0.70,
                0.75,
                0.80,
                0.85,
                0.90,
                0.95,
                1.00,
            ],
            values=[
                0.3500,
                0.3925,
                0.4350,
                0.4775,
                0.5200,
                0.5625,
                0.6050,
                0.6475,
                0.6900,
                0.7325,
                0.7750,
                0.8175,
                0.8600,
                0.9025,
                0.9450,
                0.9875,
                1.0300,
                1.0725,
                1.1150,
                1.1575,
                1.2000,
            ],
        )
        model.curves.add_curve(
            name="ThermalConductivityBlock",
            coords=[
                0.00,
                0.05,
                0.10,
                0.15,
                0.20,
                0.25,
                0.30,
                0.35,
                0.40,
                0.45,
                0.50,
                0.55,
                0.60,
                0.65,
                0.70,
                0.75,
                0.80,
                0.85,
                0.90,
                0.95,
                1.00,
            ],
            values=[
                0.260,
                0.295,
                0.330,
                0.365,
                0.400,
                0.435,
                0.470,
                0.505,
                0.540,
                0.575,
                0.610,
                0.645,
                0.680,
                0.715,
                0.750,
                0.785,
                0.820,
                0.855,
                0.890,
                0.925,
                0.960,
            ],
        )
        model.curves.add_curve(
            name="ViscosityWater",
            coords=[
                273.15,
                278.15,
                283.15,
                288.15,
                293.15,
                298.15,
                303.15,
                308.15,
                313.15,
                318.15,
                323.15,
                328.15,
                333.15,
                338.15,
                343.15,
                348.15,
                353.15,
                358.15,
                363.15,
                368.15,
                373.15,
                378.15,
                383.15,
                388.15,
                393.15,
                398.15,
                403.15,
                408.15,
                413.15,
                418.15,
                423.15,
                428.15,
                433.15,
                438.15,
                443.15,
                448.15,
                453.15,
            ],
            values=[
                0.001791443824493071,
                0.001518096315579494,
                0.001306005897987292,
                0.001137740703477269,
                0.001001761870211410,
                0.000890153572198349,
                0.000797321713362585,
                0.000719212401364518,
                0.000652823923857197,
                0.000595891721429222,
                0.000546679133154530,
                0.000503834893499928,
                0.000466293936304618,
                0.000433206989018526,
                0.000403889725629819,
                0.000377785466842696,
                0.000354437428880606,
                0.000333467809699829,
                0.000314561842102885,
                0.000297455502677900,
                0.000281925944210631,
                0.000267783979663162,
                0.000254868127533467,
                0.000243039856904272,
                0.000232179762477740,
                0.000222184466506006,
                0.000212964093284412,
                0.000204440197918893,
                0.000196544057975752,
                0.000189215256869282,
                0.000182400503210232,
                0.000176052642092791,
                0.000170129823355037,
                0.000164594798874974,
                0.000159414326452323,
                0.000154558662138820,
                0.000150001126288821,
            ],
        )
        model.process_variables.set_ic(
            compensate_non_equilibrium_initial_residuum="true",
            process_variable_name="displacement",
            components="2",
            order="1",
            initial_condition="displacement_ic",
        )
        model.process_variables.add_bc(
            process_variable_name="displacement",
            mesh="Decovalex-0-Boundary-Top-mapped-plain",
            type="Dirichlet",
            component="0",
            parameter="dirichlet",
        )
        model.process_variables.add_bc(
            process_variable_name="displacement",
            mesh="Decovalex-0-Boundary-Top-mapped-plain",
            type="Dirichlet",
            component="1",
            parameter="dirichlet",
        )
        model.process_variables.add_bc(
            process_variable_name="displacement",
            mesh="Decovalex-0-Boundary-Left-mapped-plain",
            type="Dirichlet",
            component="0",
            parameter="dirichlet",
        )
        model.process_variables.add_bc(
            process_variable_name="displacement",
            mesh="Decovalex-0-Boundary-Left-mapped-plain",
            type="Dirichlet",
            component="1",
            parameter="dirichlet",
        )
        model.process_variables.add_bc(
            process_variable_name="displacement",
            mesh="Decovalex-0-Boundary-Bottom-mapped-plain",
            type="Dirichlet",
            component="0",
            parameter="dirichlet",
        )
        model.process_variables.add_bc(
            process_variable_name="displacement",
            mesh="Decovalex-0-Boundary-Bottom-mapped-plain",
            type="Dirichlet",
            component="1",
            parameter="dirichlet",
        )
        model.process_variables.add_bc(
            process_variable_name="displacement",
            mesh="Decovalex-0-Boundary-Right-mapped-plain",
            type="Dirichlet",
            component="0",
            parameter="dirichlet",
        )
        model.process_variables.add_bc(
            process_variable_name="displacement",
            mesh="Decovalex-0-Boundary-Right-mapped-plain",
            type="Dirichlet",
            component="1",
            parameter="dirichlet",
        )
        model.process_variables.add_bc(
            process_variable_name="displacement",
            mesh="Decovalex-0-Boundary-Heater-mapped-plain",
            type="Dirichlet",
            component="0",
            parameter="dirichlet",
        )
        model.process_variables.add_bc(
            process_variable_name="displacement",
            mesh="Decovalex-0-Boundary-Heater-mapped-plain",
            type="Dirichlet",
            component="1",
            parameter="dirichlet",
        )
        model.process_variables.set_ic(
            process_variable_name="pressure",
            components="1",
            order="1",
            initial_condition="pressure_ic",
        )
        model.process_variables.set_ic(
            process_variable_name="temperature",
            components="1",
            order="1",
            initial_condition="temperature_ic",
        )
        model.process_variables.add_bc(
            process_variable_name="temperature",
            mesh="Decovalex-0-Boundary-Heater-mapped-plain",
            type="Neumann",
            parameter="heater",
        )

        model.nonlinear_solvers.add_non_lin_solver(
            name="basic_newton",
            type="Newton",
            damping=1,
            max_iter="100",
            linear_solver="linear_solver",
        )
        model.linear_solvers.add_lin_solver(
            name="linear_solver", kind="eigen", solver_type="SparseLU"
        )
        model.add_element(
            parent_xpath="./linear_solvers/linear_solver/eigen",
            tag="scaling",
            text="true",
        )
        model.add_block(
            "petsc",
            parent_xpath="./linear_solvers/linear_solver",
            taglist=["parameters"],
            textlist=[
                "-ksp_type bcgs -pc_type lu -ksp_rtol 1.e-20 -ksp_atol 1.e-18 -ksp_max_it 4000"
            ],
        )
        model.write_input()
        with Path("tunnel_ogs6py.prj").open("rb") as f:
            lines = f.readlines()
        with prj_trm_from_scratch.open("rb") as f:
            lines_ref = f.readlines()
        assert len(lines) == len(lines_ref)
        for i, line in enumerate(lines):
            if sys.platform == "win32":
                assert line.decode().translate(mapping) == lines_ref[
                    i
                ].decode().translate(mapping)
            else:
                assert line == lines_ref[i]

    def test_buildfromscratch_bhe(self) -> NoReturn:
        model = Project(output_file="HeatTransportBHE_ogs6py.prj", MKL=False)
        model.mesh.add_mesh(filename="bhe_mesh.vtu")
        model.mesh.add_mesh(filename="bhe_mesh_inflowsf.vtu")
        model.mesh.add_mesh(filename="bhe_mesh_bottomsf.vtu")
        model.mesh.add_mesh(filename="bhe_mesh_topsf.vtu")
        model.processes.set_process(
            name="HeatTransportBHE",
            type="HEAT_TRANSPORT_BHE",
            integration_order="2",
        )
        model.processes.add_process_variable(
            process_variable="process_variable",
            process_variable_name="temperature_soil",
        )
        model.processes.add_process_variable(
            process_variable="process_variable",
            process_variable_name="temperature_BHE1",
        )
        model.processes.add_bhe_type(bhe_type="2U")
        model.processes.add_bhe_component(
            comp_type="borehole", length="90", diameter="0.152"
        )
        model.processes.add_bhe_component(
            comp_type="pipes",
            inlet_diameter="0.0262",
            inlet_wall_thickness="0.0029",
            inlet_wall_thermal_conductivity="0.4",
            outlet_diameter="0.0262",
            outlet_wall_thickness="0.0029",
            outlet_wall_thermal_conductivity="0.4",
            distance_between_pipes="0.06",
            longitudinal_dispersion_length="0.001",
        )
        model.processes.add_bhe_component(
            comp_type="flow_and_temperature_control",
            type="PowerCurveConstantFlow",
            power_curve="scaled_power_curve",
            flow_rate="0.00037",
        )
        model.processes.add_bhe_component(
            comp_type="grout",
            density="1",
            porosity="0",
            specific_heat_capacity="1910000",
            thermal_conductivity="0.6",
        )
        model.processes.add_bhe_component(
            comp_type="refrigerant",
            density="1052",
            viscosity="0.0052",
            specific_heat_capacity="3795",
            thermal_conductivity="0.48",
            reference_temperature="20",
        )
        model.process_variables.set_ic(
            process_variable_name="temperature_soil",
            components="1",
            order="1",
            initial_condition="T0",
        )
        model.process_variables.add_bc(
            process_variable_name="temperature_soil",
            mesh="bhe_mesh_topsf",
            type="Dirichlet",
            parameter="T_Surface",
        )
        model.process_variables.add_bc(
            process_variable_name="temperature_soil",
            mesh="bhe_mesh_bottomsf",
            type="Neumann",
            parameter="dT_Groundsource",
        )
        model.process_variables.add_bc(
            process_variable_name="temperature_soil",
            mesh="bhe_mesh_inflowsf",
            type="Dirichlet",
            parameter="T_Inflow",
        )
        model.process_variables.set_ic(
            process_variable_name="temperature_BHE1",
            components="8",
            order="1",
            initial_condition="T0_BHE1",
        )
        model.media.add_property(
            medium_id="0",
            phase_type="Solid",
            name="specific_heat_capacity",
            type="Constant",
            value="2150000",
        )
        model.media.add_property(
            medium_id="0",
            phase_type="Solid",
            name="density",
            type="Constant",
            value="1",
        )
        model.media.add_property(
            medium_id="0",
            phase_type="AqueousLiquid",
            name="phase_velocity",
            type="Constant",
            value="0 0 0",
        )
        model.media.add_property(
            medium_id="0",
            phase_type="AqueousLiquid",
            name="specific_heat_capacity",
            type="Constant",
            value="4000",
        )
        model.media.add_property(
            medium_id="0",
            phase_type="AqueousLiquid",
            name="density",
            type="Constant",
            value="1000",
        )
        model.media.add_property(
            medium_id="0", name="porosity", type="Constant", value="0.1"
        )
        model.media.add_property(
            medium_id="0",
            name="thermal_conductivity",
            type="Constant",
            value="2.5",
        )
        model.media.add_property(
            medium_id="0",
            name="thermal_longitudinal_dispersivity",
            type="Constant",
            value="0",
        )
        model.media.add_property(
            medium_id="0",
            name="thermal_transversal_dispersivity",
            type="Constant",
            value="0",
        )
        model.media.add_property(
            medium_id="1",
            phase_type="Solid",
            name="specific_heat_capacity",
            type="Constant",
            value="1800000",
        )
        model.media.add_property(
            medium_id="1",
            phase_type="Solid",
            name="density",
            type="Constant",
            value="1",
        )
        model.media.add_property(
            medium_id="1",
            phase_type="AqueousLiquid",
            name="phase_velocity",
            type="Constant",
            value="0 2e-7 0",
        )
        model.media.add_property(
            medium_id="1",
            phase_type="AqueousLiquid",
            name="specific_heat_capacity",
            type="Constant",
            value="4000",
        )
        model.media.add_property(
            medium_id="1",
            phase_type="AqueousLiquid",
            name="density",
            type="Constant",
            value="1000",
        )
        model.media.add_property(
            medium_id="1", name="porosity", type="Constant", value="0.1"
        )
        model.media.add_property(
            medium_id="1",
            name="thermal_conductivity",
            type="Constant",
            value="2",
        )
        model.media.add_property(
            medium_id="1",
            name="thermal_longitudinal_dispersivity",
            type="Constant",
            value="0",
        )
        model.media.add_property(
            medium_id="1",
            name="thermal_transversal_dispersivity",
            type="Constant",
            value="0",
        )
        model.time_loop.add_process(
            process="HeatTransportBHE",
            nonlinear_solver_name="basic_picard",
            convergence_type="DeltaX",
            norm_type="NORM2",
            reltol="1e-5",
            time_discretization="BackwardEuler",
        )
        model.time_loop.set_stepping(
            process="HeatTransportBHE",
            type="FixedTimeStepping",
            t_initial="0",
            t_end="31536000",
            repeat="365",
            delta_t="86400",
        )
        model.time_loop.add_output(
            type="VTK",
            prefix="HTbhe_test",
            repeat="1",
            each_steps="1",
            variables=["temperature_soil", "temperature_BHE1"],
        )
        model.parameters.add_parameter(
            name="T0",
            type="MeshNode",
            mesh="bhe_mesh",
            field_name="temperature_soil",
        )
        model.parameters.add_parameter(
            name="T0_BHE1", type="Constant", values="11 11 11 11 11 11 11 11"
        )
        model.parameters.add_parameter(
            name="T_Surface",
            type="CurveScaled",
            curve="surface_temperature",
            parameter="T_CurveScaled",
        )
        model.parameters.add_parameter(
            name="T_CurveScaled", type="Constant", value="1"
        )
        model.parameters.add_parameter(
            name="T_Inflow",
            type="MeshNode",
            mesh="bhe_mesh_inflowsf",
            field_name="temperature_soil",
        )
        model.parameters.add_parameter(
            name="dT_Groundsource", type="Constant", value="0.06"
        )
        model.curves.add_curve(
            name="scaled_power_curve",
            coords=["0 1576800 31536000"],
            values=["-1600 0 -1600"],
        )
        model.curves.add_curve(
            name="surface_temperature",
            coords=["0 1576800 31536000"],
            values=["3 20 3"],
        )
        model.nonlinear_solvers.add_non_lin_solver(
            name="basic_picard",
            type="Picard",
            max_iter="500",
            linear_solver="general_linear_solver",
        )
        model.linear_solvers.add_lin_solver(
            name="general_linear_solver",
            kind="lis",
            solver_type="cg",
            precon_type="jacobi",
            max_iteration_step="100",
            error_tolerance="1e-16",
        )
        model.linear_solvers.add_lin_solver(
            name="general_linear_solver",
            kind="eigen",
            solver_type="BiCGSTAB",
            precon_type="ILUT",
            max_iteration_step="100",
            error_tolerance="1e-16",
        )
        model.linear_solvers.add_lin_solver(
            name="general_linear_solver",
            kind="petsc",
            prefix="gw",
            solver_type="cg",
            precon_type="bjacobi",
            max_iteration_step="100",
            error_tolerance="1e-16",
        )
        model.write_input()
        with Path("HeatTransportBHE_ogs6py.prj").open("rb") as f:
            lines = f.readlines()
        with prj_heat_transport.open("rb") as f:
            lines_ref = f.readlines()
        assert len(lines) == len(lines_ref)
        for i, line in enumerate(lines):
            if sys.platform == "win32":
                assert line.decode().translate(mapping) == lines_ref[
                    i
                ].decode().translate(mapping)
            else:
                assert line == lines_ref[i]

    def test_replace_text(self) -> NoReturn:
        prjfile = "tunnel_ogs6py_replace.prj"
        model = Project(input_file=prj_tunnel_trm, output_file=prjfile)
        model.replace_text("tunnel_replace", xpath="./time_loop/output/prefix")
        model.write_input()
        root = ET.parse(prjfile)
        find = root.findall("./time_loop/output/prefix")
        assert find[0].text == "tunnel_replace"

    def test_timedependenthet_param(self) -> NoReturn:
        prjfile = "timedephetparam.prj"
        model = Project(
            input_file=prj_time_dep_het_param,
            output_file=prjfile,
            verbose=True,
        )
        model.parameters.add_parameter(
            name="kappa1", type="Function", expression="1.e-12"
        )
        sides = ["left", "right"]
        for side in sides:
            model.parameters.add_parameter(
                name=f"TimeDependentDirichlet_{side}",
                type="TimeDependentHeterogeneousParameter",
                time=["0", "1180", "1200", "2000"],
                parameter_name=[
                    f"bc_{side}_ts1",
                    f"bc_{side}_ts59",
                    f"bc_{side}_ts60",
                    f"bc_{side}_ts100",
                ],
            )
        model.time_loop.set_stepping(
            process="LiquidFlow",
            type="IterationNumberBasedTimeStepping",
            t_initial=0,
            t_end=2000,
            initial_dt=10,
            minimum_dt=5,
            maximum_dt=20,
            number_iterations=[1, 3, 5, 10],
            multiplier=[1.2, 1.0, 0.9, 0.8],
        )
        model.write_input()
        with Path(prjfile).open("rb") as f:
            lines = f.readlines()
        with prj_time_dep_het_param_ref.open("rb") as f:
            lines_ref = f.readlines()
        assert len(lines) == len(lines_ref)
        for i, line in enumerate(lines):
            if sys.platform == "win32":
                assert line.decode().translate(mapping) == lines_ref[
                    i
                ].decode().translate(mapping)
            else:
                assert line == lines_ref[i]

    def test_python_st(self) -> NoReturn:
        prjfile = "python_st.prj"
        model = Project(input_file=prj_beier_sandbox, output_file=prjfile)
        model.geometry.add_geometry("beier_sandbox.gml")
        model.python_script.set_pyscript("simulationX_test.py")
        model.write_input()
        with Path(prjfile).open("rb") as f:
            lines = f.readlines()
        with prj_beier_sandbox_ref.open("rb") as f:
            lines_ref = f.readlines()
        assert len(lines) == len(lines_ref)
        for i, line in enumerate(lines):
            if sys.platform == "win32":
                assert line.decode().translate(mapping) == lines_ref[
                    i
                ].decode().translate(mapping)
            else:
                assert line == lines_ref[i]

    def test_robin_bc(self) -> NoReturn:
        prjfile = "robin_bc.prj"
        model = Project(input_file=prj_square_1e4_robin, output_file=prjfile)
        model.process_variables.add_bc(
            process_variable_name="temperature",
            geometrical_set="square_1x1_geometry",
            geometry="right",
            type="Robin",
            alpha="alpha",
            u_0="ambient_temperature",
        )
        model.write_input()
        with Path(prjfile).open("rb") as f:
            lines = f.readlines()
        with prj_square_1e4_robin_ref.open("rb") as f:
            lines_ref = f.readlines()
        assert len(lines) == len(lines_ref)
        for i, line in enumerate(lines):
            if sys.platform == "win32":
                assert line.decode().translate(mapping) == lines_ref[
                    i
                ].decode().translate(mapping)
            else:
                assert line == lines_ref[i]

    def test_staggered(self) -> NoReturn:
        prjfile = "staggered.prj"
        model = Project(input_file=prj_staggered, output_file=prjfile)
        model.processes.set_process(
            name="HM",
            type="HYDRO_MECHANICS",
            coupling_scheme="staggered",
            integration_order=3,
        )
        model.time_loop.add_global_process_coupling(
            max_iter=10,
            convergence_type="DeltaX",
            norm_type="NORM2",
            reltol=1e-10,
        )
        model.time_loop.add_global_process_coupling(
            convergence_type="DeltaX", norm_type="NORM2", reltol=1e-10
        )
        model.time_loop.add_process(
            process="HM",
            nonlinear_solver_name="basic_newton_p",
            convergence_type="DeltaX",
            norm_type="NORM2",
            abstol="1.0e-6",
            time_discretization="BackwardEuler",
        )
        model.time_loop.add_process(
            process="HM",
            nonlinear_solver_name="basic_newton_u",
            convergence_type="PerComponentDeltaX",
            norm_type="NORM2",
            abstols="1.0e-16 1.0e-16 1.0e-16",
            time_discretization="BackwardEuler",
        )
        model.time_loop.set_stepping(
            process="HM",
            process_count=0,
            type="FixedTimeStepping",
            t_initial=0,
            t_end=1,
            delta_t=1,
            repeat=1,
        )
        model.time_loop.set_stepping(
            process="HM",
            process_count=1,
            type="FixedTimeStepping",
            t_initial=0,
            t_end=1,
            delta_t=1,
            repeat=1,
        )

        model.write_input()
        with Path(prjfile).open("rb") as f:
            lines = f.readlines()
        with prj_staggered_ref.open("rb") as f:
            lines_ref = f.readlines()
        assert len(lines) == len(lines_ref)
        for i, line in enumerate(lines):
            if sys.platform == "win32":
                assert line.decode().translate(mapping) == lines_ref[
                    i
                ].decode().translate(mapping)
            else:
                assert line == lines_ref[i]

    def test_PID_controller(self) -> NoReturn:
        prjfile = "pid_timestepping.prj"
        model = Project(input_file=prj_pid_timestepping, output_file=prjfile)
        model.media.add_property(
            medium_id="0",
            name="relative_permeability",
            type="Curve",
            independent_variable="liquid_saturation",
            curve="relative_permeability",
        )
        model.time_loop.set_stepping(
            process="GW23",
            type="EvolutionaryPIDcontroller",
            t_initial=0,
            t_end=1600,
            dt_guess=0.1,
            dt_min=0.01,
            dt_max=2,
            rel_dt_min=0.01,
            rel_dt_max=4,
            tol=1.0,
        )
        model.write_input()
        with Path(prjfile).open("rb") as f:
            lines = f.readlines()
        with prj_pid_timestepping_ref.open("rb") as f:
            lines_ref = f.readlines()
        assert len(lines) == len(lines_ref)
        for i, line in enumerate(lines):
            if sys.platform == "win32":
                assert line.decode().translate(mapping) == lines_ref[
                    i
                ].decode().translate(mapping)
            else:
                assert line == lines_ref[i]

    def test_deactivate_replace(self) -> NoReturn:
        prjfile = "deactivate_replace.prj"
        model = Project(input_file=prj_pid_timestepping, output_file=prjfile)
        model.deactivate_property("viscosity", phase="AqueousLiquid")
        model.deactivate_parameter("p0")
        model.replace_parameter(
            name="p_Dirichlet_top",
            parametertype="Function",
            taglist=["expression"],
            textlist=["0.012"],
        )
        model.write_input()
        with Path(prjfile).open("rb") as f:
            lines = f.readlines()
        with prj_deactivate_replace.open("rb") as f:
            lines_ref = f.readlines()
        assert len(lines) == len(lines_ref)
        for i, line in enumerate(lines):
            if sys.platform == "win32":
                assert line.decode().translate(mapping) == lines_ref[
                    i
                ].decode().translate(mapping)
            else:
                assert line == lines_ref[i]

    def test_empty_replace(self) -> NoReturn:
        inputfile = Path(prj_tunnel_trm)
        prjfile = Path("tunnel_ogs6py_empty_replace.prj")
        model = Project(input_file=inputfile, output_file=prjfile)
        model.write_input()
        with inputfile.open("rb") as f:
            lines = f.readlines()
        with Path(prjfile).open("rb") as f:
            lines_ref = f.readlines()
        assert len(lines) == len(lines_ref)
        for i, line in enumerate(lines):
            if sys.platform == "win32":
                assert line.decode().translate(mapping) == lines_ref[
                    i
                ].decode().translate(mapping)
            else:
                assert line == lines_ref[i]

    def test_replace_phase_property(self) -> NoReturn:
        prjfile = "tunnel_ogs6py_replace.prj"
        model = Project(input_file=prj_tunnel_trm, output_file=prjfile)
        model.replace_phase_property_value(
            mediumid=0, phase="Solid", name="thermal_expansivity", value=5
        )
        model.write_input()
        root = ET.parse(prjfile)
        find = root.findall(
            "./media/medium/phases/phase[type='Solid']/properties/property[name='thermal_expansivity']/value"
        )
        assert find[0].text == "5"

    def test_replace_medium_property(self) -> NoReturn:
        prjfile = "tunnel_ogs6py_replace.prj"
        model = Project(input_file=prj_tunnel_trm, output_file=prjfile)
        model.replace_medium_property_value(
            mediumid=0, name="porosity", value=42
        )
        model.write_input()
        root = ET.parse(prjfile)
        find = root.findall(
            "./media/medium/properties/property[name='porosity']/value"
        )
        assert find[0].text == "42"

    def test_replace_parameter(self) -> NoReturn:
        prjfile = "tunnel_ogs6py_replace.prj"
        model = Project(input_file=prj_tunnel_trm, output_file=prjfile)
        model.replace_parameter_value(name="E", value=32)
        model.write_input()
        root = ET.parse(prjfile)
        find = root.findall("./parameters/parameter[name='E']/value")
        assert find[0].text == "32"

    def test_replace_mesh(self) -> NoReturn:
        prjfile = "tunnel_ogs6py_replacemesh.prj"
        model = Project(input_file=prj_tunnel_trm, output_file=prjfile)
        model.replace_mesh(
            oldmesh="tunnel_inner.vtu", newmesh="tunnel_inner_new.vtu"
        )
        model.write_input()
        root = ET.parse(prjfile)
        find = root.findall("./meshes/mesh")
        assert find[-1].text == "tunnel_inner_new.vtu"
        find = root.findall(
            "./process_variables/process_variable/boundary_conditions/boundary_condition/mesh"
        )
        assert find[0].text == "tunnel_right"
        assert find[1].text == "tunnel_left"
        assert find[2].text == "tunnel_bottom"
        assert find[3].text == "tunnel_top"
        assert find[4].text == "tunnel_right"
        assert find[5].text == "tunnel_left"
        assert find[6].text == "tunnel_top"
        assert find[7].text == "tunnel_bottom"
        assert find[9].text == "tunnel_right"
        assert find[10].text == "tunnel_left"
        assert find[11].text == "tunnel_top"
        assert find[12].text == "tunnel_bottom"
        assert find[8].text == "tunnel_inner_new"
        assert find[13].text == "tunnel_inner_new"

    def test_add_entry(self) -> NoReturn:
        prjfile = "tunnel_ogs6py_add_entry.prj"
        model = Project(input_file=prj_tunnel_trm, output_file=prjfile)
        model.add_element(tag="geometry", parent_xpath=".", text="geometry.gml")
        model.write_input()
        root = ET.parse(prjfile)
        find = root.findall("./geometry")
        assert find[0].text == "geometry.gml"

    def test_add_block(self) -> NoReturn:
        prjfile = "tunnel_ogs6py_add_block.prj"
        model = Project(input_file="tunnel_ogs6py.prj", output_file=prjfile)
        model.add_block(
            "parameter",
            parent_xpath="./parameters",
            taglist=["name", "type", "value"],
            textlist=["mu", "Constant", "0.001"],
        )
        model.write_input()
        root = ET.parse(prjfile)
        find = root.findall("./parameters/parameter[name='mu']/value")
        assert find[0].text == "0.001"

    def test_remove_element(self) -> NoReturn:
        prjfile = "tunnel_ogs6py_remove_element.prj"
        model = Project(input_file=prj_tunnel_trm, output_file=prjfile)
        model.remove_element(xpath="./parameters/parameter[name='E']")
        model.write_input()
        root = ET.parse(prjfile)
        find = root.findall("./parameters/parameter[name='E']/value")
        assert len(find) == 0

    def test_replace_block_by_include(self) -> NoReturn:
        prjfile = "tunnel_ogs6py_solid_inc.prj"
        model = Project(input_file=prj_tunnel_trm, output_file=prjfile)
        model.replace_block_by_include(
            xpath="./media/medium/phases/phase[type='Solid']",
            filename="solid.xml",
        )
        model.write_input(keep_includes=True)
        with Path(prjfile).open("rb") as f:
            lines = f.readlines()
        with Path(prj_solid_inc_ref).open("rb") as f:
            lines_ref = f.readlines()
        assert len(lines) == len(lines_ref)
        for i, line in enumerate(lines):
            if sys.platform == "win32":
                assert line.decode().translate(mapping) == lines_ref[
                    i
                ].decode().translate(mapping)
            else:
                assert line == lines_ref[i]
        with Path("solid.xml").open("rb") as f:
            lines = f.readlines()
        with prj_include_solid_ref.open("rb") as f:
            lines_ref = f.readlines()
        assert len(lines) == len(lines_ref)
        for i, line in enumerate(lines):
            if sys.platform == "win32":
                assert line.decode().translate(mapping) == lines_ref[
                    i
                ].decode().translate(mapping)
            else:
                assert line == lines_ref[i]

    def test_property_dataframe(self) -> NoReturn:
        model = Project(input_file=prj_tunnel_trm)
        p_df = model.property_dataframe()
        assert p_df.shape[0] == 12
        assert p_df.shape[1] == 5
        assert len(p_df["title"].sum()) == 228
        assert len(p_df["symbol"].sum()) == 125
        assert len(p_df["unit"].sum()) == 103
        entries = ["title", "symbol", "unit"]
        for entry in entries:
            for string in p_df[entry]:
                assert string.count("{") == string.count("}")
                assert string.count("(") == string.count(")")
                assert string.count("$") % 2 == 0
        assert (p_df["medium 0"].sum() - 2000007381.0510168) < 1e-15
        for entry in p_df["medium 1"]:
            assert entry is None

    def test_replace_property_in_include(self) -> NoReturn:
        prjfile = "tunnel_ogs6py_includetest.prj"
        model = Project(
            input_file=prj_tunnel_trm_withincludes, output_file=prjfile
        )
        model.replace_phase_property_value(
            mediumid=0, phase="Solid", name="thermal_expansivity", value=1e-3
        )
        model.write_input(keep_includes=True)
        with Path(prjfile).open("rb") as f:
            lines = f.readlines()
        with prj_tunnel_trm_withincludes.open("rb") as f:
            lines_ref = f.readlines()
        assert len(lines) == len(lines_ref)
        for i, line in enumerate(lines):
            if sys.platform == "win32":
                assert line.decode().translate(mapping) == lines_ref[
                    i
                ].decode().translate(mapping)
            else:
                assert line == lines_ref[i]
        with Path("solid_inc.xml").open("rb") as f:
            lines = f.readlines()
        with prj_include_solid.open("rb") as f:
            lines_ref = f.readlines()
        assert len(lines) == len(lines_ref)
        for i, line in enumerate(lines):
            if sys.platform == "win32":
                assert line.decode().translate(mapping) == lines_ref[
                    i
                ].decode().translate(mapping)
            else:
                assert line == lines_ref[i]

    def test_model_run(self) -> NoReturn:
        prjfile = prj_tunnel_trm
        # dummy *.SIF file
        sif_file = tempfile.NamedTemporaryFile(suffix=".sif")
        # dummy *.notSIF file
        x_file = tempfile.NamedTemporaryFile(suffix=".x")
        # dummy directory
        sing_dir = tempfile.TemporaryDirectory()

        # case: path is not a dir
        model = Project(input_file=prjfile, output_file=prjfile)
        with pytest.raises(
            RuntimeError, match=r"The specified path is not a directory.*"
        ):
            model.run_model(path="not/a/dir", container_path=sif_file.name)

        # case: container_path is not a file:
        with pytest.raises(
            RuntimeError, match=r"The specific container-path is not a file.*"
        ):
            model.run_model(container_path="not/a/file")

        # case: container_path is not a *.sif file
        with pytest.raises(
            RuntimeError,
            match=r"The specific file is not a Singularity container.*",
        ):
            model.run_model(container_path=x_file.name)

        # case Singularity executable not found without path
        if (
            shutil.which("singularity") is None
            and platform.system() != "Windows"
        ):
            with pytest.raises(
                RuntimeError,
                match=r"The Singularity executable was not found.*",
            ):
                model.run_model(container_path=sif_file.name)

        # case Singularity executable not found in path
        if platform.system() != "Windows":
            with pytest.raises(
                RuntimeError,
                match=r"The Singularity executable was not found.*",
            ):
                model.run_model(
                    path=sing_dir.name, container_path=sif_file.name
                )

        # clean up the temporary dir
        sing_dir.cleanup()
