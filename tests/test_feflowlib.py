import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

import ogstools.meshlib as ml
from ogstools import examples
from ogstools.ogs6py import Project

pytest.importorskip("ifm")

import ifm_contrib as ifm  # noqa: E402

# pylint: disable=C0413,C0412
from ogstools.feflowlib import (  # noqa: E402
    component_transport,
    convert_properties_mesh,
    extract_cell_boundary_conditions,
    extract_point_boundary_conditions,
    feflowModel,
    get_material_properties_of_CT_model,
    get_material_properties_of_HT_model,
    get_species,
    hydro_thermal,
    liquid_flow,
    points_and_cells,
    setup_prj_file,
    steady_state_diffusion,
    write_point_boundary_conditions,
)
from ogstools.feflowlib.tools import get_material_properties  # noqa: E402


def test_cli():
    subprocess.run(["feflow2ogs", "--help"], check=True)


class TestSimulation_Neumann:
    def setup_method(self):
        self.temp_dir = Path(tempfile.mkdtemp("feflow_test_simulation"))
        self.doc = ifm.loadDocument(str(examples.feflow_model_box_Neumann))
        self.pv_mesh = convert_properties_mesh(self.doc)
        neumann = np.array(self.pv_mesh["P_BCFLOW_2ND"])
        neumann = neumann[~np.isnan(neumann)]
        self.vtu_path = self.temp_dir / "boxNeumann.vtu"
        self.pv_mesh.save(self.vtu_path)
        write_point_boundary_conditions(self.temp_dir, self.pv_mesh)
        path_topsurface, topsurface = extract_cell_boundary_conditions(
            self.vtu_path, self.pv_mesh
        )
        topsurface.save(path_topsurface)

    def test_neumann_ogs_steady_state_diffusion(self):
        """
        Test if ogs simulation for a steady state diffusion results
        are similar to FEFLOW simulation results.
        """
        # Run ogs
        prjfile = self.temp_dir / "boxNeumann_test.prj"
        ssd_model = steady_state_diffusion(
            self.temp_dir / "sim_boxNeumann",
            Project(output_file=prjfile),
        )
        model = setup_prj_file(
            self.vtu_path,
            self.pv_mesh,
            get_material_properties(self.pv_mesh, "P_CONDX"),
            "steady state diffusion",
            model=ssd_model,
        )
        model.write_input()
        model.run_model(logfile=str(self.temp_dir / "out.log"))

        # Compare ogs simulation with FEFLOW simulation
        ogs_sim_res = pv.read(
            str(self.temp_dir / "sim_boxNeumann_ts_1_t_1.000000.vtu")
        )
        dif = (
            ogs_sim_res.point_data["HEAD_OGS"]
            - self.pv_mesh.point_data["P_HEAD"]
        )
        np.testing.assert_array_less(np.abs(dif), 9e-6)

    def test_neumann_ogs_liquid_flow(self):
        """
        Test if ogs simulation with liquid flow results
        are similar to FEFLOW simulation results.
        """
        # Run ogs
        prjfile = self.temp_dir / "boxNeumann_test.prj"
        lqf_model = liquid_flow(
            self.temp_dir / "sim_boxNeumann",
            Project(output_file=prjfile),
        )
        model = setup_prj_file(
            self.vtu_path,
            self.pv_mesh,
            get_material_properties(self.pv_mesh, "P_CONDX"),
            "liquid flow",
            model=lqf_model,
        )
        model.write_input()
        model.run_model(logfile=str(self.temp_dir / "out.log"))

        # Compare ogs simulation with FEFLOW simulation
        ogs_sim_res = pv.read(
            str(self.temp_dir / "sim_boxNeumann_ts_1_t_1.000000.vtu")
        )
        np.testing.assert_allclose(
            ogs_sim_res["HEAD_OGS"],
            self.pv_mesh.point_data["P_HEAD"],
            atol=5e-6,
        )


class TestSimulation_Robin:
    def setup_method(self):
        self.temp_dir = Path(tempfile.mkdtemp("feflow_test_simulation"))
        self.doc = ifm.loadDocument(str(examples.feflow_model_box_Robin))
        self.pv_mesh = convert_properties_mesh(self.doc)
        self.vtu_path = self.temp_dir / "boxRobin.vtu"
        self.pv_mesh.save(self.vtu_path)
        write_point_boundary_conditions(self.temp_dir, self.pv_mesh)
        path_topsurface, topsurface = extract_cell_boundary_conditions(
            self.vtu_path, self.pv_mesh
        )
        topsurface.save(path_topsurface)

    def test_robin_ogs_steady_state_diffusion(self):
        """
        Test if ogs simulation for a steady state diffusion results
        are similar to FEFLOW simulation results.
        """
        # Run ogs
        prjfile = self.temp_dir / "boxRobin_test.prj"
        ssd_model = steady_state_diffusion(
            str(self.temp_dir / "sim_boxRobin"),
            Project(output_file=prjfile),
        )
        model = setup_prj_file(
            self.vtu_path,
            self.pv_mesh,
            get_material_properties(self.pv_mesh, "P_CONDX"),
            "steady state diffusion",
            model=ssd_model,
        )
        model.write_input()
        model.run_model(logfile=str(self.temp_dir / "out.log"))

        # Compare ogs simulation with FEFLOW simulation
        ogs_sim_res = pv.read(
            str(self.temp_dir / "sim_boxRobin_ts_1_t_1.000000.vtu")
        )
        np.testing.assert_allclose(
            ogs_sim_res["HEAD_OGS"],
            self.pv_mesh.point_data["P_HEAD"],
            atol=6e-5,
        )

    def test_robin_ogs_liquid_flow(self):
        """
        Test if ogs simulation for a steady state diffusion results
        are similar to FEFLOW simulation results.
        """
        # Run ogs
        prjfile = self.temp_dir / "boxRobin_test.prj"
        lqf_model = liquid_flow(
            str(self.temp_dir / "sim_boxRobin"),
            Project(output_file=prjfile),
        )
        model = setup_prj_file(
            self.vtu_path,
            self.pv_mesh,
            get_material_properties(self.pv_mesh, "P_CONDX"),
            "liquid flow",
            model=lqf_model,
        )
        model.write_input()
        model.run_model(logfile=str(self.temp_dir / "out.log"))

        # Compare ogs simulation with FEFLOW simulation
        ogs_sim_res = pv.read(
            str(self.temp_dir / "sim_boxRobin_ts_1_t_1.000000.vtu")
        )
        np.testing.assert_allclose(
            ogs_sim_res["HEAD_OGS"],
            self.pv_mesh.point_data["P_HEAD"],
            atol=6e-5,
        )


class TestSimulation_Well:
    def setup_method(self):
        self.temp_dir = Path(tempfile.mkdtemp("feflow_test_simulation"))
        self.doc = ifm.loadDocument(str(examples.feflow_model_box_well_BC))
        self.pv_mesh = convert_properties_mesh(self.doc)
        self.vtu_path = self.temp_dir / "boxWell.vtu"
        self.pv_mesh.save(self.vtu_path)
        write_point_boundary_conditions(self.temp_dir, self.pv_mesh)
        path_topsurface, topsurface = extract_cell_boundary_conditions(
            self.vtu_path, self.pv_mesh
        )
        topsurface.save(path_topsurface)

    def test_well_ogs_steady_state_diffusion(self):
        """
        Test if ogs simulation for a steady state diffusion results
        are similar to FEFLOW simulation results.
        """
        # Run ogs
        prjfile = self.temp_dir / "boxWell_test.prj"
        ssd_model = steady_state_diffusion(
            str(self.temp_dir / "sim_boxWell"),
            Project(output_file=prjfile),
        )
        model = setup_prj_file(
            self.vtu_path,
            self.pv_mesh,
            get_material_properties(self.pv_mesh, "P_CONDX"),
            "steady state diffusion",
            model=ssd_model,
        )
        model.write_input()
        model.run_model(logfile=str(self.temp_dir / "out.log"))
        # Compare ogs simulation with FEFLOW simulation
        ogs_sim_res = pv.read(
            str(self.temp_dir / "sim_boxWell_ts_1_t_1.000000.vtu")
        )
        np.testing.assert_allclose(
            ogs_sim_res["HEAD_OGS"],
            self.pv_mesh.point_data["P_HEAD"],
            atol=5e-8,
        )

    def test_well_ogs_liquid_flow(self):
        """
        Test if ogs simulation for a steady state diffusion results
        are similar to FEFLOW simulation results.
        """
        # Run ogs
        prjfile = self.temp_dir / "boxWell_test.prj"
        lqf_model = liquid_flow(
            str(self.temp_dir / "sim_boxWell"),
            Project(output_file=prjfile),
        )
        model = setup_prj_file(
            self.vtu_path,
            self.pv_mesh,
            get_material_properties(self.pv_mesh, "P_CONDX"),
            "liquid flow",
            model=lqf_model,
        )
        model.write_input()
        model.run_model(logfile=str(self.temp_dir / "out.log"))

        # Compare ogs simulation with FEFLOW simulation
        ogs_sim_res = pv.read(
            str(self.temp_dir / "sim_boxWell_ts_1_t_1.000000.vtu")
        )
        np.testing.assert_allclose(
            ogs_sim_res["HEAD_OGS"],
            self.pv_mesh.point_data["P_HEAD"],
            atol=1e-10,
        )


class TestConverter:
    def setup_method(self):
        # Variables for the following tests:
        self.temp_dir = Path(tempfile.mkdtemp("feflow_test_converter"))
        self.doc = ifm.loadDocument(str(examples.feflow_model_box_Neumann))
        self.pv_mesh = convert_properties_mesh(self.doc)

    def test_mesh_class(self):
        mesh = ml.Mesh.read_feflow(examples.feflow_model_box_Neumann)
        assert mesh.n_points == 6768
        assert mesh.n_cells == 11462
        assert mesh.get_cell(0).type == pv.CellType.WEDGE

    def test_mesh_manipulation_3D(self):
        testing_mesh = self.pv_mesh.copy()
        extract_point_boundary_conditions(self.temp_dir, testing_mesh)
        assert testing_mesh.n_arrays == self.pv_mesh.n_arrays
        extract_cell_boundary_conditions(
            self.temp_dir / "boxNeumann.vtu", self.pv_mesh
        )[1]
        assert testing_mesh.n_arrays == self.pv_mesh.n_arrays

    def test_geometry(self):
        """
        Test if geometry can be converted correctly.
        """
        doc = ifm.loadDocument(str(examples.feflow_model_2layers))
        points, cells, celltypes = points_and_cells(doc)
        assert len(points) == 75
        assert len(celltypes) == 32
        assert celltypes[0] == pv.CellType.HEXAHEDRON

    def test_toymodel_mesh_conversion(self):
        """
        Test if geometry of a toymodel is converted correctly.
        """
        # 1. Test if geometry is fine
        points, cells, celltypes = points_and_cells(self.doc)
        assert len(points) == 6768
        assert len(celltypes) == 11462
        assert celltypes[0] == pv.CellType.WEDGE

        # 2. Test data arrays
        assert len(self.pv_mesh.cell_data) == 12
        assert len(self.pv_mesh.point_data) == 11

    def test_toymodel_point_boundary_condition(self):
        """
        Test if separate meshes for boundary condition are written correctly.
        """
        write_point_boundary_conditions(self.temp_dir, self.pv_mesh)
        bc_flow = pv.read(str(self.temp_dir / "P_BC_FLOW.vtu"))
        assert bc_flow.n_points == 66
        assert len(bc_flow.point_data) == 2
        bc_flow_2nd = pv.read(str(self.temp_dir / "P_BCFLOW_2ND.vtu"))
        assert bc_flow_2nd.n_points == 66
        assert len(bc_flow_2nd.point_data) == 2

    def test_toymodel_cell_boundary_condition(self):
        """
        Test if separate meshes for boundary condition are written correctly.
        """
        topsurface = extract_cell_boundary_conditions(
            self.temp_dir / "boxNeumann.vtu", self.pv_mesh
        )[1]
        cell_data_list_expected = ["P_IOFLOW", "P_SOUF", "bulk_element_ids"]
        cell_data_list = list(topsurface.cell_data)
        for cell_data, cell_data_expected in zip(
            cell_data_list, cell_data_list_expected, strict=False
        ):
            assert cell_data == cell_data_expected
        assert topsurface.n_points == 564
        assert topsurface.n_cells == 1042

    def test_toymodel_prj_file(self):
        """
        Test the prj_file that can be written
        """
        model = setup_prj_file(
            self.temp_dir / "boxNeumann_.vtu",
            self.pv_mesh,
            get_material_properties(self.pv_mesh, "P_CONDX"),
            "steady state diffusion",
        )
        model.write_input(self.temp_dir / "boxNeumann_.prj")
        prjfile_root = ET.parse(
            str(self.temp_dir / "boxNeumann_.prj")
        ).getroot()
        elements = list(prjfile_root)
        assert len(elements) == 8
        # Test if the meshes are correct
        meshes = prjfile_root.find("meshes")
        meshes_list = [mesh.text for mesh in meshes.findall("mesh")]
        meshes_list_expected = [
            "boxNeumann_.vtu",
            "topsurface_boxNeumann_.vtu",
            "P_BC_FLOW.vtu",
            "P_BCFLOW_2ND.vtu",
        ]
        for mesh, mesh_expected in zip(
            meshes_list, meshes_list_expected, strict=False
        ):
            assert mesh == mesh_expected
        # Test if the parameters are correct
        parameters = prjfile_root.find("parameters")
        parameters_list = [
            parameter.find("name").text
            for parameter in parameters.findall("parameter")
        ]
        parameters_list_expected = [
            "p0",
            "P_BC_FLOW",
            "P_BCFLOW_2ND",
            "P_IOFLOW",
            "P_SOUF",
        ]
        # Test if boundary conditions are written correctly.
        for parameter, parameter_expected in zip(
            parameters_list, parameters_list_expected, strict=False
        ):
            assert parameter == parameter_expected

        boundary_conditions = prjfile_root.find(
            "process_variables/process_variable/boundary_conditions"
        )

        boundary_condtitions_list = [
            boundary_condition.find("mesh").text
            for boundary_condition in boundary_conditions.findall(
                "boundary_condition"
            )
        ]
        for bc, bc_expected in zip(
            boundary_condtitions_list, meshes_list_expected[2:], strict=False
        ):
            assert bc == bc_expected.replace(".vtu", "")

        diffusion_value = prjfile_root.find(
            "media/medium[@id='0']/properties/property[name='diffusion']/value"
        ).text
        # The index [0] is because one needs to compare one value from the list. And all
        # values are the same.
        assert float(diffusion_value) == float(
            self.pv_mesh.cell_data["P_CONDX"][0]
        )


class TestSimulation_HT:
    def setup_method(self):
        self.temp_dir = Path(tempfile.mkdtemp("feflow_test_simulation"))
        self.doc = ifm.loadDocument(str(examples.feflow_model_2D_HT_model))
        self.pv_mesh = convert_properties_mesh(self.doc)
        self.vtu_path = self.temp_dir / "HT_Dirichlet.vtu"
        self.pv_mesh.save(self.vtu_path)
        write_point_boundary_conditions(self.temp_dir, self.pv_mesh)

    def test_mesh_manipulation_2D(self):
        testing_mesh = self.pv_mesh.copy()
        extract_point_boundary_conditions(self.temp_dir, testing_mesh)
        assert testing_mesh.n_arrays == self.pv_mesh.n_arrays
        extract_cell_boundary_conditions(
            self.temp_dir / "HT_Dirichlet_top.vtu", self.pv_mesh
        )[1]
        assert testing_mesh.n_arrays == self.pv_mesh.n_arrays

    def test_dirichlet_toymodel_ogs_ht(self):
        """
        Test if ogs simulation for a hydro thermal process results
        are equal to FEFLOW simulation results.
        """
        # Run ogs
        if self.pv_mesh.celltypes[0] in [5, 9]:
            dimension = 2
        prjfile = self.temp_dir / "HT_Dirichlet.prj"
        prj = hydro_thermal(
            str(self.temp_dir / "sim_HT_Dirichlet"),
            Project(output_file=prjfile),
            dimension,
        )
        prj = setup_prj_file(
            self.vtu_path,
            self.pv_mesh,
            get_material_properties_of_HT_model(self.pv_mesh),
            "hydro thermal",
            model=prj,
        )
        prj.write_input(prjfile)
        prj.run_model(logfile=str(self.temp_dir / "out.log"))
        # Compare ogs simulation with FEFLOW simulation
        ogs_sim_res = pv.read(
            str(
                self.temp_dir
                / "sim_HT_Dirichlet_ts_10_t_100000000000.000000.vtu"
            )
        )
        np.testing.assert_allclose(
            ogs_sim_res["temperature"], self.pv_mesh.point_data["P_TEMP"], 1e-10
        )
        np.testing.assert_allclose(
            ogs_sim_res["HEAD_OGS"], self.pv_mesh.point_data["P_HEAD"], 1e-9
        )


class TestSimulation_CT:
    def setup_method(self):
        self.temp_dir = Path(tempfile.mkdtemp("feflow_test_simulation"))
        self.doc = ifm.loadDocument(str(examples.feflow_model_2D_CT_t_560))
        self.pv_mesh_560 = convert_properties_mesh(self.doc)
        self.pv_mesh_560.save(self.temp_dir / "CT_2D_line.vtu")
        self.doc = ifm.loadDocument(str(examples.feflow_model_2D_CT_t_168))
        self.pv_mesh_168 = convert_properties_mesh(self.doc)
        self.pv_mesh_168.save(self.temp_dir / "CT_2D_line_168.vtu")
        self.doc = ifm.loadDocument(str(examples.feflow_model_2D_CT_t_28))
        self.pv_mesh_28 = convert_properties_mesh(self.doc)
        self.pv_mesh_28.save(self.temp_dir / "CT_2D_line_28.vtu")
        write_point_boundary_conditions(self.temp_dir, self.pv_mesh_560)

    def test_2_d_line_ct(self):
        """
        Test if ogs simulation for a component transport process results
        are equal to FEFLOW simulation results.
        """
        # Run ogs
        if self.pv_mesh_560.celltypes[0] in [5, 9]:
            dimension = 2

        prjfile = self.temp_dir / "CT_2D_line.prj"
        species = get_species(self.pv_mesh_560)
        prj = component_transport(
            self.temp_dir / "CT_2D_line",
            species,
            Project(output_file=prjfile),
            dimension,
            fixed_out_times=[
                2419200,
                14515200,
                48384000,
            ],
            initial_time=0,
            end_time=int(4.8384e07),
            time_stepping=list(
                zip([10] * 8, [8.64 * 10**i for i in range(8)], strict=False)
            ),
        )
        prj = setup_prj_file(
            self.temp_dir / "CT_2D_line.vtu",
            self.pv_mesh_560,
            get_material_properties_of_CT_model(self.pv_mesh_560),
            "component transport",
            species_list=species,
            model=prj,
            max_iter=6,
            rel_tol=1e-14,
        )
        prj.write_input(prjfile)
        prj.run_model(logfile=str(self.temp_dir / "out.log"))
        # Compare ogs simulation with FEFLOW simulation
        ogs_sim_res_560 = pv.read(
            str(self.temp_dir / "CT_2D_line_ts_67_t_48384000.000000.vtu")
        )
        ogs_sim_res_168 = pv.read(
            str(self.temp_dir / "CT_2D_line_ts_62_t_14515200.000000.vtu")
        )
        ogs_sim_res_28 = pv.read(
            str(self.temp_dir / "CT_2D_line_ts_52_t_2419200.000000.vtu")
        )
        # Assert concentration values:
        np.testing.assert_allclose(
            ogs_sim_res_560.point_data["single_species"],
            self.pv_mesh_560.point_data["single_species_P_CONC"],
            atol=6e-8,
        )
        np.testing.assert_allclose(
            ogs_sim_res_168.point_data["single_species"],
            self.pv_mesh_168.point_data["single_species_P_CONC"],
            atol=8e-8,
        )
        np.testing.assert_allclose(
            ogs_sim_res_28.point_data["single_species"],
            self.pv_mesh_28.point_data["single_species_P_CONC"],
            atol=2e-7,
        )
        # Assert hydraulic head:
        np.testing.assert_allclose(
            ogs_sim_res_560.point_data["HEAD_OGS"],
            self.pv_mesh_560.point_data["P_HEAD"],
            atol=1e-11,
            rtol=0.01,
            verbose=True,
        )


class TestFeflowModel:
    def setup_method(self):
        self.feflow_model = feflowModel(examples.feflow_model_2D_HT_model)

    def test_bulk_mesh(self):
        bulk_mesh = self.feflow_model.ogs_bulk_mesh
        assert bulk_mesh.n_arrays == 1

    def test_material_properties(self):
        material_prop = self.feflow_model.material_properties
        assert material_prop[0]["anisotropy_angle"] == 0
        assert material_prop[0]["specific_heat_capacity_fluid"] == 4200000

    def test_boundary_conditions(self):
        # rename method to boundary conditions
        boundary_conditions = self.feflow_model.boundary_meshes
        first_bc = boundary_conditions[next(iter(boundary_conditions))]
        assert first_bc.n_cells == 23
        assert first_bc.n_points == 23
        assert first_bc.n_arrays == 2

    def test_process(self):
        assert self.feflow_model.process == "hydro thermal"
        assert (
            feflowModel(examples.feflow_model_2D_CT_t_28).process
            == "component transport"
        )
        assert (
            feflowModel(examples.feflow_model_box_Robin).process
            == "liquid flow"
        )

    def test_prj_file(self):
        temp_dir = str(tempfile.mkdtemp("feflow_test_simulation"))
        model = feflowModel(
            examples.feflow_model_box_Neumann,
            temp_dir + "/boxNeumann_feflow_model.vtu",
        )
        model_prj = model.prj
        model_prj.write_input()
        prjfile_root = ET.parse(
            temp_dir + "/boxNeumann_feflow_model.prj"
        ).getroot()

        elements = list(prjfile_root)
        assert len(elements) == 8
        # Test if the meshes are correct
        meshes = prjfile_root.find("meshes")
        meshes_list = [mesh.text for mesh in meshes.findall("mesh")]
        meshes_list_expected = [
            "boxNeumann_feflow_model.vtu",
            "topsurface_boxNeumann_feflow_model.vtu",
            "P_BC_FLOW.vtu",
            "P_BCFLOW_2ND.vtu",
        ]
        for mesh, mesh_expected in zip(
            meshes_list, meshes_list_expected, strict=False
        ):
            assert mesh == mesh_expected
        # Test if the parameters are correct
        parameters = prjfile_root.find("parameters")
        parameters_list = [
            parameter.find("name").text
            for parameter in parameters.findall("parameter")
        ]
        parameters_list_expected = [
            "p0",
            "P_BC_FLOW",
            "P_BCFLOW_2ND",
            "P_IOFLOW",
            "P_SOUF",
        ]
        # Test if boundary conditions are written correctly.
        for parameter, parameter_expected in zip(
            parameters_list, parameters_list_expected, strict=False
        ):
            assert parameter == parameter_expected

        boundary_conditions = prjfile_root.find(
            "process_variables/process_variable/boundary_conditions"
        )

        boundary_condtitions_list = [
            boundary_condition.find("mesh").text
            for boundary_condition in boundary_conditions.findall(
                "boundary_condition"
            )
        ]
        for bc, bc_expected in zip(
            boundary_condtitions_list, meshes_list_expected[2:], strict=False
        ):
            assert bc == bc_expected.replace(".vtu", "")

        permeability = prjfile_root.find(
            "media/medium[@id='0']/properties/property[name='permeability']/value"
        ).text[0:23]
        # The index [0:23] is because one needs to read all decimals to get the value.
        assert float(permeability) == float(
            self.feflow_model.mesh.cell_data["P_COND"][0]
        )
