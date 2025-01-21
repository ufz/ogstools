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

import ifm_contrib as ifm  # noqa: E402 / because of ifm-skip

from ogstools import FeflowModel  # noqa: E402 / because of ifm-skip
from ogstools.feflowlib import (  # noqa: E402 / because of ifm-skip
    _feflowlib,  # / because of ifm-skip
    _prj_tools,  # / because of ifm-skip
    _templates,  # / because of ifm-skip
    _tools,  # / because of ifm-skip
)


def test_cli():
    subprocess.run(["feflow2ogs", "--help"], check=True)


class TestSimulation_Neumann:
    def setup_method(self):
        self.temp_dir = Path(tempfile.mkdtemp("feflow_test_simulation"))
        self.doc = ifm.loadDocument(str(examples.feflow_model_box_Neumann))
        self.pv_mesh = _feflowlib.convert_properties_mesh(self.doc)
        neumann = np.array(self.pv_mesh["P_BCFLOW_2ND"])
        neumann = neumann[~np.isnan(neumann)]
        self.vtu_path = self.temp_dir / "boxNeumann.vtu"
        self.pv_mesh.save(self.vtu_path)
        _tools.write_point_boundary_conditions(self.temp_dir, self.pv_mesh)
        path_topsurface, topsurface = _tools.extract_cell_boundary_conditions(
            self.pv_mesh
        )
        topsurface.save(path_topsurface + ".vtu")

    def test_neumann_ogs_steady_state_diffusion(self):
        """
        Test if OGS simulation results for steady state diffusion
        are similar to FEFLOW simulation results with Neumann BC.
        """
        # Run ogs
        prjfile = self.temp_dir / "boxNeumann_test.prj"
        ssd_model = _templates.steady_state_diffusion(
            self.temp_dir / "sim_boxNeumann",
            Project(output_file=prjfile),
        )
        prj = _prj_tools.setup_prj_file(
            self.vtu_path,
            self.pv_mesh,
            _tools.get_material_properties_of_H_model(self.pv_mesh),
            "Steady state diffusion",
            model=ssd_model,
        )
        prj.write_input()
        prj.run_model(logfile=str(self.temp_dir / "out.log"))

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
        Test if OGS simulation results for liquid flow process
        are similar to FEFLOW simulation results with Neumann BC.
        """
        # Run ogs
        prjfile = self.temp_dir / "boxNeumann_test.prj"
        lqf_model = _templates.liquid_flow(
            self.temp_dir / "sim_boxNeumann",
            Project(output_file=prjfile),
            end_time=int(1e8),
            time_stepping=[(1, 10), (1, 100), (1, 1000), (1, 1e6), (1, 1e7)],
        )
        prj = _prj_tools.setup_prj_file(
            self.vtu_path,
            self.pv_mesh,
            _tools.get_material_properties_of_H_model(self.pv_mesh),
            "Liquid flow",
            model=lqf_model,
        )
        prj.write_input()
        prj.run_model(logfile=str(self.temp_dir / "out.log"))

        # Compare ogs simulation with FEFLOW simulation
        ms = ml.MeshSeries(self.temp_dir / "sim_boxNeumann.pvd")
        ogs_sim_res = ms.mesh(ms.timesteps[-1])
        np.testing.assert_allclose(
            ogs_sim_res["HEAD_OGS"],
            self.pv_mesh.point_data["P_HEAD"],
            atol=5e-6,
        )


class TestSimulation_Robin:
    def setup_method(self):
        self.temp_dir = Path(tempfile.mkdtemp("feflow_test_simulation"))
        self.doc = ifm.loadDocument(str(examples.feflow_model_box_Robin))
        self.pv_mesh = _feflowlib.convert_properties_mesh(self.doc)
        self.vtu_path = self.temp_dir / "boxRobin.vtu"
        self.pv_mesh.save(self.vtu_path)
        _tools.write_point_boundary_conditions(self.temp_dir, self.pv_mesh)
        path_topsurface, topsurface = _tools.extract_cell_boundary_conditions(
            self.pv_mesh
        )
        topsurface.save(path_topsurface + ".vtu")

    def test_robin_ogs_steady_state_diffusion(self):
        """
        Test if OGS simulation results for steady state diffusion
        are similar to FEFLOW simulation results with Robin/Cauchy BC.
        """
        # Run ogs
        prjfile = self.temp_dir / "boxRobin_test.prj"
        ssd_model = _templates.steady_state_diffusion(
            str(self.temp_dir / "sim_boxRobin"),
            Project(output_file=prjfile),
        )
        prj = _prj_tools.setup_prj_file(
            self.vtu_path,
            self.pv_mesh,
            _tools.get_material_properties_of_H_model(self.pv_mesh),
            "Steady state diffusion",
            model=ssd_model,
        )
        prj.write_input()
        prj.run_model(logfile=str(self.temp_dir / "out.log"))

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
        Test if OGS simulation results for liquid flow process
        are similar to FEFLOW simulation results with Robin/Cauchy BC.
        """
        # Run ogs
        prjfile = self.temp_dir / "boxRobin_test.prj"
        lqf_model = _templates.liquid_flow(
            str(self.temp_dir / "sim_boxRobin"),
            Project(output_file=prjfile),
            end_time=int(1e8),
            time_stepping=[(1, 10), (1, 100), (1, 1000), (1, 1e6), (1, 1e7)],
        )
        prj = _prj_tools.setup_prj_file(
            self.vtu_path,
            self.pv_mesh,
            _tools.get_material_properties_of_H_model(self.pv_mesh),
            "Liquid flow",
            model=lqf_model,
        )
        prj.write_input()
        prj.run_model(logfile=str(self.temp_dir / "out.log"))

        # Compare ogs simulation with FEFLOW simulation
        ms = ml.MeshSeries(self.temp_dir / "sim_boxRobin.pvd")
        # Read the last timestep:
        ogs_sim_res = ms.mesh(ms.timesteps[-1])
        np.testing.assert_allclose(
            ogs_sim_res["HEAD_OGS"],
            self.pv_mesh.point_data["P_HEAD"],
            atol=6e-5,
        )


class TestSimulation_Well:
    def setup_method(self):
        self.temp_dir = Path(tempfile.mkdtemp("feflow_test_simulation"))
        self.doc = ifm.loadDocument(str(examples.feflow_model_box_well_BC))
        self.pv_mesh = _feflowlib.convert_properties_mesh(self.doc)
        self.vtu_path = self.temp_dir / "boxWell.vtu"
        self.pv_mesh.save(self.vtu_path)
        _tools.write_point_boundary_conditions(self.temp_dir, self.pv_mesh)
        path_topsurface, topsurface = _tools.extract_cell_boundary_conditions(
            self.pv_mesh
        )
        topsurface.save(path_topsurface + ".vtu")

    def test_well_ogs_steady_state_diffusion(self):
        """
        Test if OGS simulation results for steady state diffusion
        are similar to FEFLOW simulation results with source/sink term.
        """
        # Run ogs
        prjfile = self.temp_dir / "boxWell_test.prj"
        ssd_model = _templates.steady_state_diffusion(
            str(self.temp_dir / "sim_boxWell"),
            Project(output_file=prjfile),
        )
        prj = _prj_tools.setup_prj_file(
            self.vtu_path,
            self.pv_mesh,
            _tools.get_material_properties_of_H_model(self.pv_mesh),
            "Steady state diffusion",
            model=ssd_model,
        )
        prj.write_input()
        prj.run_model(logfile=str(self.temp_dir / "out.log"))
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
        Test if OGS simulation results for liquid flow process
        are similar to FEFLOW simulation results with source/sink term.
        """
        # Run ogs
        prjfile = self.temp_dir / "boxWell_test.prj"
        lqf_model = _templates.liquid_flow(
            str(self.temp_dir / "sim_boxWell"),
            Project(output_file=prjfile),
            end_time=int(1e8),
            time_stepping=[(1, 10), (1, 100), (1, 1000), (1, 1e6), (1, 1e7)],
        )
        prj = _prj_tools.setup_prj_file(
            self.vtu_path,
            self.pv_mesh,
            _tools.get_material_properties_of_H_model(self.pv_mesh),
            "Liquid flow",
            model=lqf_model,
        )
        prj.write_input()
        prj.run_model(logfile=str(self.temp_dir / "out.log"))

        # Compare ogs simulation with FEFLOW simulation
        ms = ml.MeshSeries(self.temp_dir / "sim_boxWell.pvd")
        # Read the last timestep:
        ogs_sim_res = ms.mesh(ms.timesteps[-1])
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
        self.pv_mesh = _feflowlib.convert_properties_mesh(self.doc)

    def test_mesh_class(self):
        "Test if ogstools-mesh class can read FEFLOW model."
        mesh = ml.Mesh.read_feflow(examples.feflow_model_box_Neumann)
        assert mesh.n_points == 6768
        assert mesh.n_cells == 11462
        assert mesh.get_cell(0).type == pv.CellType.WEDGE

    def test_mesh_preservation_3D(self):
        "Test if converted properties mesh preserves unchanged after extraction of BC."
        testing_mesh = self.pv_mesh.copy()
        _tools.extract_point_boundary_conditions(testing_mesh)
        assert testing_mesh.n_arrays == self.pv_mesh.n_arrays
        _tools.extract_cell_boundary_conditions(self.pv_mesh)[1]
        assert testing_mesh.n_arrays == self.pv_mesh.n_arrays

    def test_geometry(self):
        "Test if geometry can be converted correctly."
        doc = ifm.loadDocument(str(examples.feflow_model_2D_HT))
        points, cells, celltypes = _feflowlib.points_and_cells(doc)
        assert len(points) == 3228
        assert len(celltypes) == 6260
        assert celltypes[0] == pv.CellType.TRIANGLE

    def test_toymodel_mesh_conversion(self):
        "Test if geometry of a toymodel is converted correctly."
        # 1. Test if geometry is fine
        points, cells, celltypes = _feflowlib.points_and_cells(self.doc)
        assert len(points) == 6768
        assert len(celltypes) == 11462
        assert celltypes[0] == pv.CellType.WEDGE

        # 2. Test data arrays
        assert len(self.pv_mesh.cell_data) == 12
        assert len(self.pv_mesh.point_data) == 11

    def test_toymodel_point_boundary_condition(self):
        "Test if separate meshes for boundary condition are written correctly."
        _tools.write_point_boundary_conditions(self.temp_dir, self.pv_mesh)
        bc_flow = pv.read(str(self.temp_dir / "P_BC_FLOW.vtu"))
        assert bc_flow.n_points == 66
        assert len(bc_flow.point_data) == 2
        bc_flow_2nd = pv.read(str(self.temp_dir / "P_BCFLOW_2ND.vtu"))
        assert bc_flow_2nd.n_points == 66
        assert len(bc_flow_2nd.point_data) == 2

    def test_toymodel_cell_boundary_condition(self):
        "Test if separate meshes for boundary condition are written correctly."
        topsurface = _tools.extract_cell_boundary_conditions(self.pv_mesh)[1]
        cell_data_list_expected = ["P_IOFLOW", "P_SOUF", "bulk_element_ids"]
        cell_data_list = list(topsurface.cell_data)
        for cell_data, cell_data_expected in zip(
            cell_data_list, cell_data_list_expected, strict=False
        ):
            assert cell_data == cell_data_expected
        assert topsurface.n_points == 564
        assert topsurface.n_cells == 1042

    def test_toymodel_prj_file(self):
        "Test the prj_file that can be written."
        prj = _prj_tools.setup_prj_file(
            self.temp_dir / "boxNeumann.vtu",
            self.pv_mesh,
            _tools.get_material_properties_of_H_model(self.pv_mesh),
            "Steady state diffusion",
        )
        prj.write_input(self.temp_dir / "boxNeumann.prj")
        prjfile_root = ET.parse(str(self.temp_dir / "boxNeumann.prj")).getroot()
        elements = list(prjfile_root)
        assert len(elements) == 8
        # Test if the meshes are correct
        meshes = prjfile_root.find("meshes")
        meshes_list = [mesh.text for mesh in meshes.findall("mesh")]
        meshes_list_expected = [
            "boxNeumann.vtu",
            "P_BC_FLOW.vtu",
            "P_BCFLOW_2ND.vtu",
        ]

        assert meshes_list == meshes_list_expected
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
            boundary_condtitions_list, meshes_list_expected[1:], strict=False
        ):
            assert bc == bc_expected.replace(".vtu", "")

        diffusion_value = prjfile_root.find(
            "media/medium[@id='0']/properties/property[name='diffusion']/value"
        ).text[0:23]
        # The index [0:23] is because one needs to read all decimals to get the value.
        assert float(diffusion_value) == float(
            self.pv_mesh.cell_data["P_CONDX"][0]
        )


class TestSimulation_HT:
    def setup_method(self):
        self.temp_dir = Path(tempfile.mkdtemp("feflow_test_simulation"))
        self.doc = ifm.loadDocument(str(examples.feflow_model_2D_HT))
        self.pv_mesh = _feflowlib.convert_properties_mesh(self.doc)
        self.vtu_path = self.temp_dir / "HT_Dirichlet.vtu"
        self.pv_mesh.save(self.vtu_path)
        _tools.write_point_boundary_conditions(self.temp_dir, self.pv_mesh)

    def test_mesh_preservation_2D(self):
        "Test if converted properties mesh preserves unchanged after extraction of BC."
        testing_mesh = self.pv_mesh.copy()
        _tools.extract_point_boundary_conditions(testing_mesh)
        assert testing_mesh.n_arrays == self.pv_mesh.n_arrays
        _tools.extract_cell_boundary_conditions(self.pv_mesh)[1]
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
        prj = _templates.hydro_thermal(
            str(self.temp_dir / "sim_HT_Dirichlet"),
            Project(output_file=prjfile),
            dimension,
            end_time=1e11,
            time_stepping=[(1, 1e10)],
        )
        prj = _prj_tools.setup_prj_file(
            self.vtu_path,
            self.pv_mesh,
            _tools.get_material_properties_of_HT_model(self.pv_mesh),
            "Hydro thermal",
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
            ogs_sim_res["temperature"], self.pv_mesh.point_data["P_TEMP"], 8e-9
        )
        np.testing.assert_allclose(
            ogs_sim_res["HEAD_OGS"], self.pv_mesh.point_data["P_HEAD"], 1e-8
        )


class TestSimulation_CT:
    def setup_method(self):
        self.temp_dir = Path(tempfile.mkdtemp("feflow_test_simulation"))
        self.doc = ifm.loadDocument(str(examples.feflow_model_2D_CT_t_560))
        self.pv_mesh_560 = _feflowlib.convert_properties_mesh(self.doc)
        self.pv_mesh_560.save(self.temp_dir / "CT_2D_line.vtu")
        self.doc = ifm.loadDocument(str(examples.feflow_model_2D_CT_t_168))
        self.pv_mesh_168 = _feflowlib.convert_properties_mesh(self.doc)
        self.pv_mesh_168.save(self.temp_dir / "CT_2D_line_168.vtu")
        self.doc = ifm.loadDocument(str(examples.feflow_model_2D_CT_t_28))
        self.pv_mesh_28 = _feflowlib.convert_properties_mesh(self.doc)
        self.pv_mesh_28.save(self.temp_dir / "CT_2D_line_28.vtu")
        _tools.write_point_boundary_conditions(self.temp_dir, self.pv_mesh_560)

    def test_2_d_line_ct(self):
        """
        Test if ogs simulation for a component transport process results
        are equal to FEFLOW simulation results.
        """
        # Run ogs
        if self.pv_mesh_560.celltypes[0] in [5, 9]:
            dimension = 2

        prjfile = self.temp_dir / "CT_2D_line.prj"
        species = _tools.get_species(self.pv_mesh_560)
        prj = _templates.component_transport(
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
        prj = _prj_tools.setup_prj_file(
            self.temp_dir / "CT_2D_line.vtu",
            self.pv_mesh_560,
            _tools.get_material_properties_of_CT_model(self.pv_mesh_560),
            "Component transport",
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
        self.temp_dir = Path(tempfile.mkdtemp("feflow_test_simulation"))
        self.feflow_model_HT = FeflowModel(
            examples.feflow_model_2D_HT, self.temp_dir / "HT"
        )
        self.feflow_model_HT_hetero = FeflowModel(
            examples.feflow_model_2D_HT_hetero, self.temp_dir / "HT_hetero"
        )
        self.feflow_model_HTC = FeflowModel(
            examples.feflow_model_2D_HTC, self.temp_dir / "HTC"
        )
        self.feflow_model_H = FeflowModel(
            examples.feflow_model_box_Neumann, self.temp_dir / "boxNeumann"
        )
        self.feflow_H_box_IOFLOW = FeflowModel(examples.feflow_model_box_IOFLOW)

    def test_H_model_LQF_SSD(self):
        """
        Test if converted FeflowModel object can be run to reproduce FEFLOW results
        for liquid flow and steady state diffusion.
        """
        self.feflow_model_H.setup_prj(
            end_time=int(1e8),
            time_stepping=[(1, 10), (1, 100), (1, 1000), (1, 1e6), (1, 1e7)],
        )
        self.feflow_model_H.run()
        ms = ml.MeshSeries(self.temp_dir / "boxNeumann.pvd")
        ogs_sim_res = ms.mesh(ms.timesteps[-1])
        np.testing.assert_allclose(
            ogs_sim_res["HEAD_OGS"],
            self.feflow_model_H.mesh.point_data["P_HEAD"],
            atol=5e-6,
        )
        self.feflow_model_H.setup_prj(steady=True)
        self.feflow_model_H.run()
        ms = ml.MeshSeries(self.temp_dir / "boxNeumann.pvd")
        ogs_sim_res = ms.mesh(ms.timesteps[-1])
        np.testing.assert_allclose(
            ogs_sim_res["HEAD_OGS"],
            self.feflow_model_H.mesh.point_data["P_HEAD"],
            atol=5e-6,
        )

    def test_HT_model(self):
        """
        Test if converted FeflowModel object can be run hydro thermal process.
        """
        self.feflow_model_HT.setup_prj(
            end_time=int(1e11),
            time_stepping=[(1, 1e10)],
        )
        self.feflow_model_HT.run()
        ms = ml.MeshSeries(self.temp_dir / "HT.pvd")
        ogs_sim_res = ms.mesh(ms.timesteps[-1])

        np.testing.assert_allclose(
            ogs_sim_res["HEAD_OGS"],
            self.feflow_model_HT.mesh.point_data["P_HEAD"],
            atol=1e-9,
        )

        np.testing.assert_allclose(
            ogs_sim_res["temperature"],
            self.feflow_model_HT.mesh.point_data["P_TEMP"],
            atol=2,
            rtol=5e-05,
        )

    def test_HT_model_heterogeneous_material_properties(self):
        """
        Test if converted FeflowModel object can be run hydro thermal process with heterogeneous material properties.
        Also Test if prj-file is correct.
        """
        self.feflow_model_HT_hetero.setup_prj(
            end_time=int(1e11),
            time_stepping=[(1, 1e10)],
        )
        self.feflow_model_HT_hetero.run()
        ms = ml.MeshSeries(self.temp_dir / "HT_hetero.pvd")
        ogs_sim_res = ms.mesh(ms.timesteps[-1])

        """
        Head diff is too big, probably need better configuration.
        np.testing.assert_allclose(
            ogs_sim_res["HEAD_OGS"],
            self.feflow_model_HT_hetero.mesh.point_data["P_HEAD"],
            atol=1e-9,
        )
        """

        np.testing.assert_allclose(
            ogs_sim_res["temperature"],
            self.feflow_model_HT_hetero.mesh.point_data["P_TEMP"],
            atol=2,
            rtol=5e-05,
        )
        prjfile_root = ET.parse(self.temp_dir / "HT_hetero.prj").getroot()

        elements = list(prjfile_root)
        assert len(elements) == 8
        parameters = prjfile_root.find("parameters")
        parameters_list = [
            parameter.find("name").text
            for parameter in parameters.findall("parameter")
        ]
        parameters_list_expected = [
            "T0",
            "p0",
            "P_BC_FLOW",
            "P_BCFLOW_2ND",
            "P_BC_HEAT",
            "P_COND",
            "P_CONDUCF",
            "P_POROH",
        ]
        for parameter, parameter_expected in zip(
            parameters_list, parameters_list_expected, strict=False
        ):
            assert parameter == parameter_expected

    def test_bulk_mesh(self):
        "Test if bulk mesh only contains 1 array (MaterialIDs)."
        bulk_mesh = self.feflow_model_HT.ogs_bulk_mesh
        assert bulk_mesh.n_arrays == 1

        bulk_mesh = self.feflow_model_H.ogs_bulk_mesh
        assert bulk_mesh.n_arrays == 1

        bulk_mesh = self.feflow_model_HT_hetero.ogs_bulk_mesh
        assert bulk_mesh.n_arrays == 1

        bulk_mesh = self.feflow_model_HTC.ogs_bulk_mesh
        assert bulk_mesh.n_arrays == 1

    def test_material_properties(self):
        "Test if material properties are returned correctly from FeflowModel"
        material_prop = self.feflow_model_HT.material_properties
        material_ID = 0
        assert material_prop[material_ID]["anisotropy_angle"] == 0
        assert material_prop[material_ID]["anisotropy_factor"] == 1
        assert material_prop[material_ID]["storage"] == 0
        assert (
            material_prop[material_ID]["permeability"] == 1.1574074074074073e-05
        )
        assert material_prop[material_ID]["thermal_conductivity_fluid"] == 0.65
        assert material_prop[material_ID]["thermal_conductivity_solid"] == 3
        assert material_prop[material_ID]["porosity"] == 0.4000000059604645
        assert (
            material_prop[material_ID]["thermal_longitudinal_dispersivity"] == 5
        )
        assert (
            material_prop[material_ID]["thermal_transversal_dispersivity"]
            == 0.5
        )

        assert (
            material_prop[material_ID]["specific_heat_capacity_fluid"]
            == 4200000
        )
        assert (
            material_prop[material_ID]["specific_heat_capacity_solid"]
            == 1633000
        )

        material_prop_hetero = self.feflow_model_HT_hetero.material_properties
        assert (
            "inhomogeneous" in material_prop_hetero[material_ID]["permeability"]
        )
        assert "inhomogeneous" in material_prop_hetero[material_ID]["porosity"]
        assert "inhomogeneous" in (
            material_prop_hetero[material_ID]["thermal_conductivity_fluid"]
        )
        assert (
            material_prop_hetero[material_ID]["thermal_conductivity_solid"]
            == 3.0
        )

        material_prop_H = self.feflow_model_H.material_properties
        assert (
            material_prop_H[material_ID]["permeability_X"]
            == 1.1574074074074073e-05
        )
        assert material_prop_H[material_ID]["storage"] == 9.999999747378752e-05

        assert (
            "not supported"
            in self.feflow_model_HTC.material_properties["undefined"][0]
        )

    def test_boundary_conditions(self):
        "Test for one model (HT) if boundary condition are returned correctly from FeflowModel."
        boundary_conditions = self.feflow_model_HT.subdomains
        first_bc = boundary_conditions[next(iter(boundary_conditions))]
        assert first_bc.n_cells == 44
        assert first_bc.n_points == 44
        assert first_bc.n_arrays == 2
        boundary_conditions = self.feflow_model_HT_hetero.subdomains
        neumann_BC = boundary_conditions["P_BCFLOW_2ND"]
        assert neumann_BC.celltypes[0] == 3  # 3 is a Line element
        assert "topsurface" in self.feflow_H_box_IOFLOW.subdomains

    def test_processes(self):
        "Test if processes are detected correctly."
        assert self.feflow_model_HT.process == "Hydro thermal"
        assert (
            FeflowModel(examples.feflow_model_2D_CT_t_28).process
            == "Component transport"
        )
        assert (
            FeflowModel(examples.feflow_model_box_Robin).process
            == "Liquid flow"
        )
        assert (
            "not supported by this feflow converter"
            in self.feflow_model_HTC.process
        )

    def test_prj_file_HT(self):
        """
        Test if prj-file is created correctly using
        FeflowModel object for a HT process.
        """
        temp_dir = Path(tempfile.mkdtemp("feflow_test_simulation"))
        model = FeflowModel(
            examples.feflow_model_box_Neumann,
            temp_dir / "boxNeumann_feflow_model",
        )
        model.project.write_input()
        prjfile_root = ET.parse(
            temp_dir / "boxNeumann_feflow_model.prj"
        ).getroot()

        elements = list(prjfile_root)
        assert len(elements) == 8
        # Test if the meshes are correct
        meshes = prjfile_root.find("meshes")
        meshes_list = [mesh.text for mesh in meshes.findall("mesh")]
        meshes_list_expected = [
            "boxNeumann_feflow_model.vtu",
            "P_BC_FLOW.vtu",
            "P_BCFLOW_2ND.vtu",
        ]
        assert meshes_list == meshes_list_expected
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
            boundary_condtitions_list, meshes_list_expected[1:], strict=False
        ):
            assert bc == bc_expected.replace(".vtu", "")

        permeability = prjfile_root.find(
            "media/medium[@id='0']/properties/property[name='permeability']/value"
        ).text[0:23]
        # The index [0:23] is because one needs to read all decimals to get the value.
        assert float(permeability) == float(
            self.feflow_model_HT.mesh.cell_data["P_COND"][0]
        )  # cell_data["P_COND"][0] refers to the first value of the "P_COND" array,
        # as the values are homogeneous they are all the same.

    def test_prj_file_HTC(self):
        """
        Test if prj-file is created correctly using
        FeflowModel object for a HTC process.
        """
        model = self.feflow_model_HTC
        model.project.write_input()
        prjfile_root = ET.parse(str(self.temp_dir / "HTC.prj")).getroot()

        elements = list(prjfile_root)
        assert len(elements) == 8
        # Test if the meshes are correct
        meshes = prjfile_root.find("meshes")
        meshes_list = [mesh.text for mesh in meshes.findall("mesh")]
        meshes_list_expected = [
            "HTC.vtu",
            "P_BC_FLOW.vtu",
            "P_BCFLOW_2ND.vtu",
            "single_species_P_BC_MASS.vtu",
        ]
        assert meshes_list == meshes_list_expected
        # Test if the parameters are correct

        parameters = prjfile_root.find("parameters")
        parameters_list = [
            parameter.find("name").text
            for parameter in parameters.findall("parameter")
        ]
        parameters_list_expected = [
            "T0",
            "C0",
            "p0",
            "P_BC_FLOW",
            "P_BCFLOW_2ND",
            "single_species_P_BC_MASS",
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
        bc_expected_list = [
            "single_species_P_BC_MASS",
            "P_BC_FLOW",
            "P_BCFLOW_2ND",
        ]
        for bc, bc_expected in zip(
            boundary_condtitions_list, bc_expected_list, strict=False
        ):
            assert bc == bc_expected
