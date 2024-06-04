import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import pyvista as pv
from ogs6py import ogs

from ogstools import examples

pytest.importorskip("ifm")

import ifm_contrib as ifm  # noqa: E402

from ogstools.feflowlib import (  # noqa: E402
    component_transport,
    convert_properties_mesh,
    extract_cell_boundary_conditions,
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
from ogstools.feflowlib.tools import (  # noqa: E402
    get_material_properties,
)


def test_cli():
    subprocess.run(["feflow2ogs", "--help"], check=True)


class TestSimulation_Neumann(unittest.TestCase):
    def setUp(self):
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

    def test_Neumann_ogs_steady_state_diffusion(self):
        """
        Test if ogs simulation for a steady state diffusion results
        are similar to FEFLOW simulation results.
        """
        # Run ogs
        prjfile = self.temp_dir / "boxNeumann_test.prj"
        ssd_model = steady_state_diffusion(
            self.temp_dir / "sim_boxNeumann",
            ogs.OGS(PROJECT_FILE=prjfile),
        )
        model = setup_prj_file(
            self.vtu_path,
            self.pv_mesh,
            get_material_properties(self.pv_mesh, "P_CONDX"),
            "steady state diffusion",
            model=ssd_model,
        )
        model.write_input(prjfile)
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

    def test_Neumann_ogs_liquid_flow(self):
        """
        Test if ogs simulation with liquid flow results
        are similar to FEFLOW simulation results.
        """
        # Run ogs
        prjfile = self.temp_dir / "boxNeumann_test.prj"
        lqf_model = liquid_flow(
            self.temp_dir / "sim_boxNeumann",
            ogs.OGS(PROJECT_FILE=prjfile),
        )
        model = setup_prj_file(
            self.vtu_path,
            self.pv_mesh,
            get_material_properties(self.pv_mesh, "P_CONDX"),
            "liquid flow",
            model=lqf_model,
        )
        model.write_input(prjfile)
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


class TestSimulation_Robin(unittest.TestCase):
    def setUp(self):
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

    def test_Robin_ogs_steady_state_diffusion(self):
        """
        Test if ogs simulation for a steady state diffusion results
        are similar to FEFLOW simulation results.
        """
        # Run ogs
        prjfile = self.temp_dir / "boxRobin_test.prj"
        ssd_model = steady_state_diffusion(
            str(self.temp_dir / "sim_boxRobin"),
            ogs.OGS(PROJECT_FILE=prjfile),
        )
        model = setup_prj_file(
            self.vtu_path,
            self.pv_mesh,
            get_material_properties(self.pv_mesh, "P_CONDX"),
            "steady state diffusion",
            model=ssd_model,
        )
        model.write_input(prjfile)
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

    def test_Robin_ogs_liquid_flow(self):
        """
        Test if ogs simulation for a steady state diffusion results
        are similar to FEFLOW simulation results.
        """
        # Run ogs
        prjfile = self.temp_dir / "boxRobin_test.prj"
        lqf_model = liquid_flow(
            str(self.temp_dir / "sim_boxRobin"),
            ogs.OGS(PROJECT_FILE=prjfile),
        )
        model = setup_prj_file(
            self.vtu_path,
            self.pv_mesh,
            get_material_properties(self.pv_mesh, "P_CONDX"),
            "liquid flow",
            model=lqf_model,
        )
        model.write_input(prjfile)
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


class TestSimulation_Well(unittest.TestCase):
    def setUp(self):
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

    def test_Well_ogs_steady_state_diffusion(self):
        """
        Test if ogs simulation for a steady state diffusion results
        are similar to FEFLOW simulation results.
        """
        # Run ogs
        prjfile = self.temp_dir / "boxWell_test.prj"
        ssd_model = steady_state_diffusion(
            str(self.temp_dir / "sim_boxWell"),
            ogs.OGS(PROJECT_FILE=prjfile),
        )
        model = setup_prj_file(
            self.vtu_path,
            self.pv_mesh,
            get_material_properties(self.pv_mesh, "P_CONDX"),
            "steady state diffusion",
            model=ssd_model,
        )
        model.write_input(prjfile)
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

    def test_Well_ogs_liquid_flow(self):
        """
        Test if ogs simulation for a steady state diffusion results
        are similar to FEFLOW simulation results.
        """
        # Run ogs
        prjfile = self.temp_dir / "boxWell_test.prj"
        lqf_model = liquid_flow(
            str(self.temp_dir / "sim_boxWell"),
            ogs.OGS(PROJECT_FILE=prjfile),
        )
        model = setup_prj_file(
            self.vtu_path,
            self.pv_mesh,
            get_material_properties(self.pv_mesh, "P_CONDX"),
            "liquid flow",
            model=lqf_model,
        )
        model.write_input(prjfile)
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


class TestConverter(unittest.TestCase):
    def setUp(self):
        # Variables for the following tests:
        self.temp_dir = Path(tempfile.mkdtemp("feflow_test_converter"))
        self.doc = ifm.loadDocument(str(examples.feflow_model_box_Neumann))
        self.pv_mesh = convert_properties_mesh(self.doc)

    def test_geometry(self):
        """
        Test if geometry can be converted correctly.
        """
        doc = ifm.loadDocument(str(examples.feflow_model_2layers))
        points, cells, celltypes = points_and_cells(doc)
        self.assertEqual(len(points), 75)
        self.assertEqual(len(celltypes), 32)
        self.assertEqual(celltypes[0], pv.CellType.HEXAHEDRON)

    def test_toymodel_mesh_conversion(self):
        """
        Test if geometry of a toymodel is converted correctly.
        """
        # 1. Test if geometry is fine
        points, cells, celltypes = points_and_cells(self.doc)
        self.assertEqual(len(points), 6768)
        self.assertEqual(len(celltypes), 11462)
        self.assertEqual(celltypes[0], pv.CellType.WEDGE)

        # 2. Test data arrays
        self.assertEqual(len(self.pv_mesh.cell_data), 12)
        self.assertEqual(len(self.pv_mesh.point_data), 11)

    def test_toymodel_point_boundary_condition(self):
        """
        Test if separate meshes for boundary condition are written correctly.
        """
        write_point_boundary_conditions(self.temp_dir, self.pv_mesh)
        bc_flow = pv.read(str(self.temp_dir / "P_BC_FLOW.vtu"))
        self.assertEqual(bc_flow.n_points, 66)
        self.assertEqual(len(bc_flow.point_data), 2)
        bc_flow_2nd = pv.read(str(self.temp_dir / "P_BCFLOW_2ND.vtu"))
        self.assertEqual(bc_flow_2nd.n_points, 66)
        self.assertEqual(len(bc_flow_2nd.point_data), 2)

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
            cell_data_list, cell_data_list_expected
        ):
            self.assertEqual(cell_data, cell_data_expected)
        self.assertEqual(topsurface.n_points, 564)
        self.assertEqual(topsurface.n_cells, 1042)

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
        self.assertEqual(len(elements), 8)
        # Test if the meshes are correct
        meshes = prjfile_root.find("meshes")
        meshes_list = [mesh.text for mesh in meshes.findall("mesh")]
        meshes_list_expected = [
            "boxNeumann_.vtu",
            "topsurface_boxNeumann_.vtu",
            "P_BC_FLOW.vtu",
            "P_BCFLOW_2ND.vtu",
        ]
        for mesh, mesh_expected in zip(meshes_list, meshes_list_expected):
            self.assertEqual(mesh, mesh_expected)
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
            parameters_list, parameters_list_expected
        ):
            self.assertEqual(parameter, parameter_expected)

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
            boundary_condtitions_list, meshes_list_expected[2:]
        ):
            self.assertEqual(bc, bc_expected.replace(".vtu", ""))

        diffusion_value = prjfile_root.find(
            "media/medium[@id='0']/properties/property[name='diffusion']/value"
        ).text
        # The index [0] is because one needs to compare one value from the list. And all
        # values are the same.
        self.assertEqual(
            float(diffusion_value),
            float(self.pv_mesh.cell_data["P_CONDX"][0]),
        )


class TestSimulation_HT(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp("feflow_test_simulation"))
        self.doc = ifm.loadDocument(str(examples.feflow_model_2D_HT_model))
        self.pv_mesh = convert_properties_mesh(self.doc)
        self.vtu_path = self.temp_dir / "HT_Dirichlet.vtu"
        self.pv_mesh.save(self.vtu_path)
        write_point_boundary_conditions(self.temp_dir, self.pv_mesh)

    def test_Dirichlet_toymodel_ogs_HT(self):
        """
        Test if ogs simulation for a hydro thermal process results
        are equal to FEFLOW simulation results.
        """
        # Run ogs
        if self.pv_mesh.celltypes[0] in [5, 9]:
            dimension = 2
        prjfile = self.temp_dir / "HT_Dirichlet.prj"
        model = hydro_thermal(
            str(self.temp_dir / "sim_HT_Dirichlet"),
            ogs.OGS(PROJECT_FILE=prjfile),
            dimension,
        )
        model = setup_prj_file(
            self.vtu_path,
            self.pv_mesh,
            get_material_properties_of_HT_model(self.pv_mesh),
            "hydro thermal",
            model=model,
        )
        model.write_input(prjfile)
        model.run_model(logfile=str(self.temp_dir / "out.log"))
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


class TestSimulation_CT(unittest.TestCase):
    def setUp(self):
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

    def test_2D_line_CT(self):
        """
        Test if ogs simulation for a component transport process results
        are equal to FEFLOW simulation results.
        """
        # Run ogs
        if self.pv_mesh_560.celltypes[0] in [5, 9]:
            dimension = 2

        prjfile = self.temp_dir / "CT_2D_line.prj"
        species = get_species(self.pv_mesh_560)
        model = component_transport(
            self.temp_dir / "CT_2D_line",
            species,
            ogs.OGS(PROJECT_FILE=prjfile),
            dimension,
            fixed_out_times=[
                2419200,
                14515200,
                48384000,
            ],
        )
        model = setup_prj_file(
            self.temp_dir / "CT_2D_line.vtu",
            self.pv_mesh_560,
            get_material_properties_of_CT_model(self.pv_mesh_560),
            "component transport",
            species_list=species,
            model=model,
            initial_time=0,
            end_time=4.8384e07,
            time_stepping=list(
                zip([10] * 8, [8.64 * 10**i for i in range(8)])
            ),
            max_iter=6,
            rel_tol=1e-14,
        )
        model.write_input(prjfile)
        model.run_model(logfile=str(self.temp_dir / "out.log"))
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


if __name__ == "__main__":
    unittest.main(argv=[""], verbosity=2, exit=False)
