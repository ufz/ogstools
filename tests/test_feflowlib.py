import subprocess
import unittest
import xml.etree.ElementTree as ET
from os import chdir
from pathlib import Path

import numpy as np
import pytest
import pyvista as pv
from ogs6py import ogs

pytest.importorskip("ifm")

import ifm_contrib as ifm  # noqa: E402

from ogstools.feflowlib import (  # noqa: E402
    read_properties,
    write_point_boundary_conditions,
)
from ogstools.feflowlib.feflowlib import points_and_cells  # noqa: E402
from ogstools.feflowlib.tools import (  # noqa: E402
    include_xml_snippet_in_prj_file,
)


def test_cli():
    subprocess.run(["feflow2ogs", "--help"], check=True)


current_dir = Path(__file__).parent


class TestConverter(unittest.TestCase):
    def test_converter(self):
        doc = ifm.loadDocument(
            str(Path(current_dir / "data/feflowlib/2layers_model.fem"))
        )
        points, cells, celltypes = points_and_cells(doc)
        assert len(points) == 75
        assert len(celltypes) == 32
        assert celltypes[0] == pv.CellType.HEXAHEDRON


class TestToymodel(unittest.TestCase):
    def test_toymodel(self):
        chdir("tests/data/feflowlib/")
        doc = ifm.loadDocument(str(Path("box_3D_neumann.fem")))

        # 1. Test if geometry is fine
        points, cells, celltypes = points_and_cells(doc)
        assert len(points) == 6768
        assert len(celltypes) == 11462
        assert celltypes[0] == pv.CellType.WEDGE

        # 2. Test if data on mesh is fine
        pv_mesh = read_properties(doc)
        assert len(pv_mesh.cell_data) == 12
        assert len(pv_mesh.point_data) == 11
        pv_mesh.save("boxNeumann.vtu")

        # 3. Test boundary condition and xml-writing
        # 3.1 Test xml-writing
        write_point_boundary_conditions("boxNeumann", pv_mesh)
        mesh_tree = ET.parse("mesh_boxNeumann.xml")
        BC_tree = ET.parse("BC_boxNeumann.xml")
        parameter_tree = ET.parse("parameter_boxNeumann.xml")
        assert mesh_tree.getroot().tag == "meshes"
        assert BC_tree.getroot().tag == "boundary_conditions"
        assert parameter_tree.getroot().tag == "parameters"

        # 3.2 Test boundary mesh-files
        bc_flow = pv.read("P_BC_FLOW.vtu")
        assert bc_flow.n_points == 66
        assert len(bc_flow.point_data) == 2
        bc_flow_2nd = pv.read("P_BCFLOW_2ND.vtu")
        assert bc_flow_2nd.n_points == 66
        assert len(bc_flow_2nd.point_data) == 2

        # 4.Run ogs simulation with ogs6py and compare Simulation results.
        include_xml_snippet_in_prj_file(
            "inBoxNeumann.prj", "boxNeumann.prj", "parameter_boxNeumann.xml"
        )

        model = ogs.OGS(
            PROJECT_FILE="boxNeumann2.prj", INPUT_FILE="boxNeumann.prj"
        )
        model.add_include(parent_xpath=".", file="mesh_boxNeumann.xml")
        model.add_include(
            parent_xpath="./process_variables/process_variable",
            file="BC_boxNeumann.xml",
        )
        model.write_input(keep_includes=True)
        model.run_model()

        # 4.1 Compare ogs simulation with FEFLOW simulation
        ogs_sim_res = pv.read("xxx_ts_2_t_86400.000000.vtu")
        dif = ogs_sim_res.point_data["pressure"] + pv_mesh.point_data["P_HEAD"]
        assert np.all(np.abs(dif) < 5e-6)
        assert np.allclose(dif, 0, atol=5e-6, rtol=0)


if __name__ == "__main__":
    unittest.main(argv=[""], verbosity=2, exit=False)
