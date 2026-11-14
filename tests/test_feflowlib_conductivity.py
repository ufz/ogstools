from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

import ogstools as ot
from ogstools.feflowlib import _prj_tools, _templates, _tools
from ogstools.feflowlib._conductivity import _conductivity_tensor


@pytest.fixture
def planar_mesh():
    mesh = pv.Plane(i_resolution=4, j_resolution=4).cast_to_unstructured_grid()
    mesh.cell_data["MaterialIDs"] = np.zeros(mesh.n_cells, dtype=np.int32)
    mesh.cell_data["P_COND"] = np.full(mesh.n_cells, 10.0)
    mesh.cell_data["P_ANIS"] = np.full(mesh.n_cells, 0.1)
    mesh.cell_data["P_ANGL"] = np.full(mesh.n_cells, 45.0)
    mesh.cell_data["P_COMP"] = np.zeros(mesh.n_cells)
    mesh.cell_data["P_INACTIVE_ELE"] = np.ones(mesh.n_cells)
    return mesh


@pytest.mark.parametrize(
    ("angle", "ratio", "expected"),
    [
        (0, 0.1, [10, 0, 0, 1]),
        (90, 0.1, [1, 0, 0, 10]),
        (45, 0.1, [5.5, 4.5, 4.5, 5.5]),
        (-45, 0.1, [5.5, -4.5, -4.5, 5.5]),
        (225, 0.1, [5.5, 4.5, 4.5, 5.5]),
        (37, 1, [10, 0, 0, 10]),
    ],
)
def test_tensor_rotation(planar_mesh, angle, ratio, expected):
    planar_mesh.cell_data["P_ANGL"][:] = angle
    planar_mesh.cell_data["P_ANIS"][:] = ratio
    actual = _conductivity_tensor(planar_mesh)
    # Round-off from evaluating sine and cosine, including at right angles.
    np.testing.assert_allclose(
        actual,
        np.tile(expected, (planar_mesh.n_cells, 1)),
        rtol=1e-14,
        atol=1e-14,
    )


@pytest.mark.parametrize("process", ["Liquid flow", "Hydro thermal"])
@pytest.mark.parametrize("steady", [False, True])
def test_constant_tensor_project(planar_mesh, process, steady):
    properties = _tools.material_properties_of_process(planar_mesh, process)
    project = ot.Project()
    _prj_tools._add_permeabilty_prj(properties, project, 0, steady)
    name = "diffusion" if steady else "permeability"
    prop = project.tree.find(
        f"./media/medium/properties/property[name='{name}']"
    )
    assert prop.findtext("type") == "Constant"
    np.testing.assert_allclose(
        np.fromstring(prop.findtext("value"), sep=" "),
        [5.5, 4.5, 4.5, 5.5],
        rtol=1e-14,
        atol=1e-14,
    )


@pytest.mark.parametrize("process", ["Liquid flow", "Hydro thermal"])
@pytest.mark.parametrize("varying", ["P_COND", "P_ANIS", "P_ANGL"])
def test_element_tensors_round_trip(planar_mesh, process, varying, tmp_path):
    # Two heterogeneous materials share one mesh parameter. Only one input
    # varies, so the other tensor components must still be retained.
    planar_mesh.cell_data["MaterialIDs"][8:] = 3
    planar_mesh.cell_data[varying][::2] *= 0.5
    properties = _tools.material_properties_of_process(planar_mesh, process)
    mesh = _tools._add_heterogeneous_material_property_to_mesh(
        planar_mesh, properties
    )
    project = ot.Project()
    for material_id in properties:
        _prj_tools._add_permeabilty_prj(properties, project, material_id)
    parameters = project.tree.findall("./parameters/parameter")
    assert len(parameters) == 1
    assert parameters[0].findtext("type") == "MeshElement"
    field = parameters[0].findtext("field_name")
    assert field == "conductivity_tensor"
    assert (
        len(
            project.tree.findall(
                ".//property[parameter_name='conductivity_tensor']"
            )
        )
        == 2
    )
    path = tmp_path / "tensor.vtu"
    mesh.save(path)
    actual = pv.read(path).cell_data[field].reshape(-1, 2, 2)
    # The two known principal conductivities are invariant under rotation.
    expected = np.column_stack(
        (mesh["P_COND"] * mesh["P_ANIS"], mesh["P_COND"])
    )
    np.testing.assert_allclose(
        np.linalg.eigvalsh(actual), expected, rtol=1e-14, atol=1e-14
    )
    np.testing.assert_allclose(
        actual, actual.transpose(0, 2, 1), rtol=0, atol=0
    )


def test_scalar_without_anisotropy(planar_mesh):
    del planar_mesh.cell_data["P_ANIS"]
    properties = _tools.get_material_properties_of_H_model(planar_mesh)
    assert "permeability_tensor" not in properties[0]
    project = ot.Project()
    _prj_tools._add_permeabilty_prj(properties, project, 0)
    assert (
        float(project.tree.findtext(".//property[name='permeability']/value"))
        == 10
    )


def test_volume_conductivity_unchanged():
    mesh = pv.ImageData(dimensions=(2, 2, 2)).cast_to_unstructured_grid()
    for name, value in {
        "MaterialIDs": 0,
        "P_CONDX": 10.0,
        "P_CONDY": 2.0,
        "P_CONDZ": 1.0,
    }.items():
        mesh.cell_data[name] = [value]
    properties = _tools.get_material_properties_of_H_model(mesh)
    assert "permeability_tensor" not in properties[0]
    project = ot.Project()
    _prj_tools._add_permeabilty_prj(properties, project, 0)
    assert (
        project.tree.findtext(".//property[name='permeability']/value")
        == "10.0 2.0 1.0"
    )


@pytest.mark.system
@pytest.mark.parametrize("heterogeneous", [False, True])
def test_rotated_tensor_flow(planar_mesh, heterogeneous, tmp_path):
    """An affine head has a known transverse flux, even across horizontal layers."""
    if heterogeneous:
        # Kxx changes across y, while Kyx remains 4.5: h=x is still a solution.
        upper = planar_mesh.cell_centers().points[:, 1] > 0
        planar_mesh.cell_data["P_COND"][upper] = 11
        planar_mesh.cell_data["P_ANIS"][upper] = 2 / 11
    properties = _tools.get_material_properties_of_H_model(planar_mesh)
    mesh = _tools._add_heterogeneous_material_property_to_mesh(
        planar_mesh, properties
    )
    project = _templates.steady_state_diffusion(
        Path("flow"), ot.Project(), dimension=2
    )
    project = _prj_tools.setup_prj_file(
        Path("domain.vtu"),
        mesh,
        properties,
        "Steady state diffusion",
        model=project,
    )
    boundary = mesh.extract_feature_edges().cast_to_unstructured_grid()
    boundary.point_data["head"] = boundary.points[:, 0].astype(float)
    project.mesh.add_mesh("boundary.vtu")
    project.parameters.add_parameter(
        name="boundary_head",
        type="MeshNode",
        mesh="boundary",
        field_name="head",
    )
    project.process_variables.add_bc(
        process_variable_name="HEAD_OGS",
        type="Dirichlet",
        mesh="boundary",
        component=0,
        parameter="boundary_head",
    )
    for variable in ["HEAD_OGS", "v"]:
        project.add_element(
            parent_xpath="./time_loop/output/variables",
            tag="variable",
            text=variable,
        )
    mesh.save(tmp_path / "domain.vtu")
    boundary.save(tmp_path / "boundary.vtu")
    simulation = ot.Model(project, meshes=tmp_path).run(tmp_path / "simulation")
    assert simulation.status == simulation.Status.done, simulation.status_str
    result = ot.MeshSeries(simulation.result.sim_output / "flow.pvd")[-1]
    np.testing.assert_allclose(
        result["HEAD_OGS"], result.points[:, 0], rtol=0, atol=1e-9
    )
    # A scalar or diagonal export can reproduce h=x, but loses this transverse flux.
    np.testing.assert_allclose(result["v"][:, 1], -4.5, rtol=0, atol=1e-8)
