import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import pyvista as pv
from ogs6py import ogs

pytest.importorskip("ifm")

import ifm_contrib as ifm  # noqa: E402

from ogstools.feflowlib import (  # noqa: E402
    convert_properties_mesh,
    extract_cell_boundary_conditions,
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


current_dir = Path(__file__).parent


class TestSimulation_Neumann(unittest.TestCase):
    def setUp(self):
        self.path_data = current_dir / "data/feflowlib/"
        self.path_writing = Path(tempfile.mkdtemp("feflow_test_simulation"))
        self.doc = ifm.loadDocument(str(self.path_data / "box_3D_neumann.fem"))
        self.pv_mesh = convert_properties_mesh(self.doc)
        neumann = np.array(self.pv_mesh["P_BCFLOW_2ND"])
        neumann = neumann[~np.isnan(neumann)]
        self.pv_mesh.save(str(self.path_writing / "boxNeumann.vtu"))
        write_point_boundary_conditions(self.path_writing, self.pv_mesh)
        path_topsurface, topsurface = extract_cell_boundary_conditions(
            self.path_writing / "boxNeumann.vtu", self.pv_mesh
        )
        topsurface.save(path_topsurface)

    def test_Neumann_ogs_steady_state_diffusion(self):
        """
        Test if ogs simulation for a steady state diffusion results
        are similar to FEFLOW simulation results.
        """
        # Run ogs
        prjfile = str(self.path_writing / "boxNeumann_test.prj")
        model = steady_state_diffusion(
            self.path_writing / "sim_boxNeumann",
            ogs.OGS(PROJECT_FILE=prjfile),
        )
        model = setup_prj_file(
            self.path_writing / "boxNeumann.vtu",
            self.pv_mesh,
            get_material_properties(self.pv_mesh, "P_CONDX"),
            "steady state diffusion",
            model,
        )
        model.write_input(prjfile)
        model.run_model(logfile=str(self.path_writing / "out.log"))

        # Compare ogs simulation with FEFLOW simulation
        ogs_sim_res = pv.read(
            str(self.path_writing / "sim_boxNeumann_ts_1_t_1.000000.vtu")
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
        prjfile = str(self.path_writing / "boxNeumann_test.prj")
        model = liquid_flow(
            self.path_writing / "sim_boxNeumann",
            ogs.OGS(PROJECT_FILE=prjfile),
        )
        setup_prj_file(
            self.path_writing / "boxNeumann.vtu",
            self.pv_mesh,
            get_material_properties(self.pv_mesh, "P_CONDX"),
            "liquid flow",
            model,
        )
        model.write_input(prjfile)
        model.run_model(logfile=str(self.path_writing / "out.log"))

        # Compare ogs simulation with FEFLOW simulation
        ogs_sim_res = pv.read(
            str(self.path_writing / "sim_boxNeumann_ts_1_t_1.000000.vtu")
        )
        dif = (
            ogs_sim_res.point_data["HEAD_OGS"]
            - self.pv_mesh.point_data["P_HEAD"]
        )
        np.testing.assert_array_less(np.abs(dif), 9e-6)


class TestSimulation_Robin(unittest.TestCase):
    def setUp(self):
        self.path_data = current_dir / "data/feflowlib/"
        self.path_writing = Path(tempfile.mkdtemp("feflow_test_simulation"))
        self.doc = ifm.loadDocument(
            str(self.path_data / "box_3D_cauchy_areal.fem")
        )
        self.pv_mesh = convert_properties_mesh(self.doc)
        self.pv_mesh.save(str(self.path_writing / "boxRobin.vtu"))
        write_point_boundary_conditions(self.path_writing, self.pv_mesh)
        path_topsurface, topsurface = extract_cell_boundary_conditions(
            self.path_writing / "boxRobin.vtu", self.pv_mesh
        )
        topsurface.save(path_topsurface)

    def test_Robin_ogs_steady_state_diffusion(self):
        """
        Test if ogs simulation for a steady state diffusion results
        are similar to FEFLOW simulation results.
        """
        # Run ogs
        prjfile = str(self.path_writing / "boxRobin_test.prj")
        model = steady_state_diffusion(
            str(self.path_writing / "sim_boxRobin"),
            ogs.OGS(PROJECT_FILE=prjfile),
        )
        model = setup_prj_file(
            self.path_writing / "boxRobin.vtu",
            self.pv_mesh,
            get_material_properties(self.pv_mesh, "P_CONDX"),
            "steady state diffusion",
            model,
        )
        model.write_input(prjfile)
        model.run_model(logfile=str(self.path_writing / "out.log"))

        # Compare ogs simulation with FEFLOW simulation
        ogs_sim_res = pv.read(
            str(self.path_writing / "sim_boxRobin_ts_1_t_1.000000.vtu")
        )
        dif = (
            ogs_sim_res.point_data["HEAD_OGS"]
            - self.pv_mesh.point_data["P_HEAD"]
        )
        np.testing.assert_array_less(np.abs(dif), 9e-5)

    def test_Robin_ogs_liquid_flow(self):
        """
        Test if ogs simulation for a steady state diffusion results
        are similar to FEFLOW simulation results.
        """
        # Run ogs
        prjfile = str(self.path_writing / "boxRobin_test.prj")
        model = liquid_flow(
            str(self.path_writing / "sim_boxRobin"),
            ogs.OGS(PROJECT_FILE=prjfile),
        )
        model = setup_prj_file(
            self.path_writing / "boxRobin.vtu",
            self.pv_mesh,
            get_material_properties(self.pv_mesh, "P_CONDX"),
            "liquid flow",
            model,
        )
        model.write_input(prjfile)
        model.run_model(logfile=str(self.path_writing / "out.log"))

        # Compare ogs simulation with FEFLOW simulation
        ogs_sim_res = pv.read(
            str(self.path_writing / "sim_boxRobin_ts_1_t_1.000000.vtu")
        )
        dif = (
            ogs_sim_res.point_data["HEAD_OGS"]
            - self.pv_mesh.point_data["P_HEAD"]
        )
        np.testing.assert_array_less(np.abs(dif), 9e-5)


class TestSimulation_Well(unittest.TestCase):
    def setUp(self):
        self.path_data = current_dir / "data/feflowlib/"
        self.path_writing = Path(tempfile.mkdtemp("feflow_test_simulation"))
        self.doc = ifm.loadDocument(str(self.path_data / "box_3D_wellBC.fem"))
        self.pv_mesh = convert_properties_mesh(self.doc)
        self.pv_mesh.save(str(self.path_writing / "boxWell.vtu"))
        write_point_boundary_conditions(self.path_writing, self.pv_mesh)
        path_topsurface, topsurface = extract_cell_boundary_conditions(
            self.path_writing / "boxWell.vtu", self.pv_mesh
        )
        topsurface.save(path_topsurface)

    def test_Well_ogs_steady_state_diffusion(self):
        """
        Test if ogs simulation for a steady state diffusion results
        are similar to FEFLOW simulation results.
        """
        # Run ogs
        prjfile = str(self.path_writing / "boxWell_test.prj")
        model = steady_state_diffusion(
            str(self.path_writing / "sim_boxWell"),
            ogs.OGS(PROJECT_FILE=prjfile),
        )
        model = setup_prj_file(
            self.path_writing / "boxWell.vtu",
            self.pv_mesh,
            get_material_properties(self.pv_mesh, "P_CONDX"),
            "steady state diffusion",
            model,
        )
        model.write_input(prjfile)
        model.run_model(logfile=str(self.path_writing / "out.log"))
        # Compare ogs simulation with FEFLOW simulation
        ogs_sim_res = pv.read(
            str(self.path_writing / "sim_boxWell_ts_1_t_1.000000.vtu")
        )
        dif = (
            ogs_sim_res.point_data["HEAD_OGS"]
            - self.pv_mesh.point_data["P_HEAD"]
        )
        np.testing.assert_array_less(np.abs(dif), 9e-8)

    def test_Well_ogs_liquid_flow(self):
        """
        Test if ogs simulation for a steady state diffusion results
        are similar to FEFLOW simulation results.
        """
        # Run ogs
        prjfile = str(self.path_writing / "boxWell_test.prj")
        model = liquid_flow(
            str(self.path_writing / "sim_boxWell"),
            ogs.OGS(PROJECT_FILE=prjfile),
        )
        model = setup_prj_file(
            self.path_writing / "boxWell.vtu",
            self.pv_mesh,
            get_material_properties(self.pv_mesh, "P_CONDX"),
            "liquid flow",
            model,
        )
        model.write_input(prjfile)
        model.run_model(logfile=str(self.path_writing / "out.log"))

        # Compare ogs simulation with FEFLOW simulation
        ogs_sim_res = pv.read(
            str(self.path_writing / "sim_boxWell_ts_1_t_1.000000.vtu")
        )
        dif = (
            ogs_sim_res.point_data["HEAD_OGS"]
            - self.pv_mesh.point_data["P_HEAD"]
        )
        np.testing.assert_array_less(np.abs(dif), 1e-9)


class TestConverter(unittest.TestCase):
    def setUp(self):
        # Variables for the following tests:
        self.path_data = current_dir / "data/feflowlib/"
        self.path_writing = Path(tempfile.mkdtemp("feflow_test_converter"))
        self.doc = ifm.loadDocument(str(self.path_data / "box_3D_neumann.fem"))
        self.pv_mesh = convert_properties_mesh(self.doc)

    def test_geometry(self):
        """
        Test if geometry can be converted correctly.
        """
        doc = ifm.loadDocument(
            str(current_dir / "data/feflowlib/2layers_model.fem")
        )
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
        write_point_boundary_conditions(self.path_writing, self.pv_mesh)
        bc_flow = pv.read(str(self.path_writing / "P_BC_FLOW.vtu"))
        self.assertEqual(bc_flow.n_points, 66)
        self.assertEqual(len(bc_flow.point_data), 2)
        bc_flow_2nd = pv.read(str(self.path_writing / "P_BCFLOW_2ND.vtu"))
        self.assertEqual(bc_flow_2nd.n_points, 66)
        self.assertEqual(len(bc_flow_2nd.point_data), 2)

    def test_toymodel_cell_boundary_condition(self):
        """
        Test if separate meshes for boundary condition are written correctly.
        """
        topsurface = extract_cell_boundary_conditions(
            self.path_writing / "boxNeumann.vtu", self.pv_mesh
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
            self.path_writing / "boxNeumann_.vtu",
            self.pv_mesh,
            get_material_properties(self.pv_mesh, "P_CONDX"),
            "steady state diffusion",
        )
        model.write_input(self.path_writing / "boxNeumann_.prj")
        prjfile_root = ET.parse(
            str(self.path_writing / "boxNeumann_.prj")
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
        for parameter, parameter_expected in zip(
            parameters_list, parameters_list_expected
        ):
            self.assertEqual(parameter, parameter_expected)
        """
        boundary_conditions = root.find('process_variables/process_variable/boundary_conditions')
        boundary_condtitions_list = [boundary_condition.find('parameter').text for boundary_condition in boundary_conditions.findall('parameter')]
        """
        diffusion_value = prjfile_root.find(
            "media/medium[@id='0']/properties/property[name='diffusion']/value"
        ).text
        # The index [0] is because I need to compare one value from the list. And all
        # values are the same.
        self.assertEqual(
            float(diffusion_value),
            float(self.pv_mesh.cell_data["P_CONDX"][0]),
        )


if __name__ == "__main__":
    unittest.main(argv=[""], verbosity=2, exit=False)
