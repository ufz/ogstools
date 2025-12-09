"""Unit tests for meshlib."""

from pathlib import Path

import numpy as np
import pytest
import pyvista as pv
from hypothesis import HealthCheck, Verbosity, example, given, settings
from hypothesis import strategies as st

import ogstools as ot
from ogstools import examples
from ogstools.definitions import EXAMPLES_DIR


@pytest.mark.parametrize("threshold_angle", [None, 20.0])
@pytest.mark.parametrize("angle_y", [0.0, -20.0, -45.0, -70.0])
def test_meshes_from_mesh(threshold_angle: None | float, angle_y: float):
    mesh = examples.load_meshseries_THM_2D_PVD()[0].rotate_y(angle_y)
    boundaries = ot.Meshes.from_mesh(mesh, threshold_angle)
    # not recommended, but here we are only interested in the boundaries
    boundaries.pop("domain")

    assert len(boundaries) == 4
    np.testing.assert_array_equal(
        [mesh.n_cells for mesh in boundaries.values()], [83, 83, 44, 44]
    )
    assert np.all([mesh.n_cells > 1 for mesh in boundaries.values()])
    assert np.all(boundaries["left"].points[:, 0] == mesh.bounds[0])
    assert np.all(boundaries["right"].points[:, 0] == mesh.bounds[1])
    assert boundaries["bottom"].bounds[2] == mesh.bounds[2]
    assert boundaries["top"].bounds[3] == mesh.bounds[3]


@pytest.mark.system()
def test_meshes_from_mesh_run(tmp_path):
    "Test using extracted boundaries for a simulation."
    mesh_path = tmp_path / "mesh.msh"
    ot.gmsh_tools.rect(n_edge_cells=(2, 4), out_name=mesh_path)
    domain = ot.Meshes.from_gmsh(mesh_path)["domain"]
    # this is no good practice and only done for testing purposes
    # we recommend to define the boundaries as physical groups within gmsh
    meshes = ot.Meshes.from_mesh(domain)
    meshes.save(tmp_path)

    model = ot.Project(
        input_file=examples.prj_mechanics,
        output_file=tmp_path / "default.prj",
    )
    model.write_input()
    model.run_model(write_logs=False, args=f"-m {tmp_path} -o {tmp_path}")
    # check for correct bulk id mapping during extraction
    mesh = ot.MeshSeries(tmp_path / "mesh.pvd")[-1]
    top_right = mesh.find_closest_point([1.0, 1.0, 0.0])
    max_uy = np.max(mesh["displacement"][:, 1])
    assert max_uy == mesh["displacement"][top_right, 1]


@st.composite
def meshing(draw: st.DrawFn):
    dim = draw(st.integers(min_value=2, max_value=3))
    mesh_func = ot.gmsh_tools.rect if dim == 2 else ot.gmsh_tools.cuboid
    n_cells = draw(st.tuples(*([st.integers(1, 3)] * dim)))
    n_layers = draw(st.integers(min_value=1, max_value=4))
    rand_id = draw(st.integers(min_value=0, max_value=n_layers - 1))
    return mesh_func, n_cells, n_layers, rand_id


@pytest.mark.tools()
@example(meshing_data=(ot.gmsh_tools.rect, (2, 2), 2, 0), failcase=True).xfail(
    # CLI version fails and doesn't write the new file, thus cannot be read
    raises=FileNotFoundError
)
@given(meshing_data=meshing(), failcase=st.just(False))
@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    verbosity=Verbosity.normal,
    deadline=None,
)
def test_identify_subdomains(tmp_path, meshing_data, failcase):
    "Testing parity between py and C++ (CLI) implementation."

    mesh_func, n_cells, n_layers, rand_id = meshing_data
    path = Path(tmp_path / f"{mesh_func.__name__}_{n_layers}_{n_cells}")
    path.mkdir(parents=True, exist_ok=True)
    mesh_name = path / "rect.msh"
    mesh_func(n_edge_cells=n_cells, n_layers=n_layers, out_name=mesh_name)
    meshes = ot.Meshes.from_gmsh(mesh_name, log=False)
    layer: pv.UnstructuredGrid
    layer = meshes["domain"].threshold(
        [rand_id, rand_id], scalars="MaterialIDs"
    )
    # multi-dim test
    if meshes["domain"].volume:
        meshes["layer_surface"] = layer.extract_surface()
    meshes["layer_edges"] = layer.extract_feature_edges()
    meshes["layer_points"] = layer.extract_points(
        range(layer.n_points), include_cells=False
    )
    if failcase:
        meshes["extra_mesh"] = meshes["layer_points"].translate([-1, 0, 0])
    for m in meshes.values():
        m.point_data.pop("bulk_node_ids", None)
        m.cell_data.pop("bulk_element_ids", None)

    serial_sub_paths = meshes.save(path, overwrite=True)
    domain_mesh = meshes.domain

    ot.cli().identifySubdomains(
        f=True,
        o=path / "new_",
        m=domain_mesh.filepath,
        *serial_sub_paths,  # noqa: B026
    )
    from ogstools.meshes.subdomains import identify_subdomains

    # actually meshes.subdomains, but here we let the domain mesh also get bulk ids
    identify_subdomains(domain_mesh, meshes.values())

    def _check(mesh_1, mesh_2, key: str):
        np.testing.assert_array_equal(mesh_1[key], mesh_2[key])

    for name, mesh in meshes.items():
        cli_subdomain = pv.read(path / f"new_{name}.vtu")
        _check(mesh, cli_subdomain, "bulk_node_ids")
        if "number_bulk_elements" in mesh.cell_data:
            _check(mesh, cli_subdomain, "number_bulk_elements")
            # order of the bulk element ids is random for multiple ids per
            # sub cell, thus only testing for equality of the set.
            cell_id = 0
            for num_cells in mesh["number_bulk_elements"]:
                view = slice(cell_id, cell_id + num_cells)
                np.testing.assert_array_equal(
                    set(mesh["bulk_element_ids"][view]),
                    set(cli_subdomain["bulk_element_ids"][view]),
                )
                cell_id += num_cells

        else:
            _check(mesh, cli_subdomain, "bulk_element_ids")


def test_meshes_saving_reading(tmp_path):
    "Check, that saving+reading meshes equal the original."
    ot.gmsh_tools.rect(out_name=tmp_path / "mesh.msh")
    meshes = ot.Meshes.from_gmsh(tmp_path / "mesh.msh", log=False)
    meshes.save(tmp_path)  # serial mesh only
    for name in meshes:
        mesh = ot.mesh.read(tmp_path / f"{name}.vtu")
        for data in ["point_data", "cell_data", "field_data"]:
            np.testing.assert_array_equal(
                getattr(mesh, data).values(),
                getattr(meshes[name], data).values(),
            )


@pytest.mark.tools()  # partmesh
@pytest.mark.parametrize("partition", [None, 1, 2, 4])
@pytest.mark.parametrize("dry_run", [False, True])
def test_meshes_save_parallel(tmp_path, partition, dry_run):
    """
    Test object: Meshes.save()
    Use Case: Stores Meshes object and optionally performs partitioning
    Assumes: A Meshes object is present. The folder is already present and empty.
    Checks: Return value of test object and that these files are existing.
    """
    "Checks the number of saved files"
    ot.gmsh_tools.rect(out_name=tmp_path / "mesh.msh")
    # additional clean folder (no gmsh file inside)
    meshes_path = Path(tmp_path / "meshes")
    meshes_path.mkdir()
    meshes = ot.Meshes.from_gmsh(tmp_path / "mesh.msh", log=False)
    files = meshes.save(meshes_path, num_partitions=partition, dry_run=dry_run)
    if partition:
        f1 = files[1]
        # Mesh contains domain, left, right, top, bottom
        assert len(f1) == 5  # checking the serial mesh
        # Each boundary (4*) 8 and domain 6
        if partition > 1:
            assert len(files[partition]) == 38
        else:  # partition==1
            assert len(files[partition]) == 5
    else:  # partition == None
        assert len(files) == 5

    if dry_run:
        assert not any(meshes_path.iterdir())  # still empty folder
    else:
        if partition:
            partition_files = [file for lst in files.values() for file in lst]
        else:
            partition_files = files
        for file in partition_files:
            assert file.exists()


@pytest.mark.parametrize("partition", [1, 2, 4])
@pytest.mark.parametrize("default_metis", [True, False])
@pytest.mark.parametrize("dry_run", [True, False])
def test_meshes_partmesh_file_only(tmp_path, partition, default_metis, dry_run):
    """
    Test object: lower level: Meshes.partmesh_prepare() and Meshes.partmesh()
    Use Case: Meshes files are already present and only partitioning is requested
    Assumes: The input meshes contain at least one point and one cell property
    Checks: If partition files are generated.
    """
    # Setup
    meshes1_path = Path(tmp_path / "meshes1")
    meshes1_path.mkdir()

    ot.gmsh_tools.rect(out_name=meshes1_path / "mesh.msh")
    meshes1 = ot.Meshes.from_gmsh(meshes1_path / "mesh.msh", log=False)

    files = meshes1.save(meshes1_path)

    meshes2 = Path(tmp_path / "meshes2")
    meshes2.mkdir()
    # End of setup

    basefile = ot.Meshes.create_metis(
        domain_file=files[0], output_path=meshes2, dry_run=dry_run
    )

    if dry_run:
        assert not basefile.exists()
    else:
        assert basefile.exists()

    if default_metis:
        basefile = None

    files = ot.Meshes.create_partitioning(
        partition, files[0], files[1:], metis_file=basefile, dry_run=dry_run
    )

    if partition == 1:
        assert len(files) == 5  # 4 subdomains + domain
    else:
        assert len(files) == 38  # subdomains(4)*8 + 6(domain)

    for file in files:
        if dry_run:
            assert not file.exists()
        else:
            assert file.exists()


def test_meshes_rename(tmp_path):
    """
    Test object:    Meshes.rename_subdomains() and Meshes.rename_subdomains_legacy()
    Use Case:       Already existing Meshes object, but file names (e.g. of .save) must follow specific naming
                    e.g. to fit into existing prj description
    Assumes:        Meshes contains  "left" and "right" named meshes
    Checks:         rename_subdomains_legacy works like rename_subdomains(+physical_group)
    """
    ot.gmsh_tools.rect(out_name=tmp_path / "mesh.msh")
    meshes = ot.Meshes.from_gmsh(tmp_path / "mesh.msh", log=False)
    left_mesh = meshes["left"]
    right_mesh = meshes["right"]
    meshes.rename_subdomains({"left": "left_new_name"})
    meshes.domain_name = "my_new_domain_name"

    assert meshes.domain_name == "my_new_domain_name"
    assert meshes["left_new_name"] == left_mesh

    with pytest.raises(KeyError, match="Invalid subdomain names"):
        meshes.rename_subdomains({"does_not_exist": "foo"})

    meshes.rename_subdomains_legacy()
    assert meshes["physical_group_right"] == right_mesh

    meshes.modify_names(prefix="prefix_", suffix="_suffix")
    assert meshes["prefix_physical_group_right_suffix"] == right_mesh


def test_meshes_from_prj(tmp_path: Path):
    "Check, that the mesh paths generated from a Project are correct."
    ot.gmsh_tools.rect(out_name=tmp_path / "mesh.msh")
    meshes_ref = ot.Meshes.from_gmsh(tmp_path / "mesh.msh", log=False)
    meshes_ref.save(tmp_path)
    prj = ot.Project(examples.prj_mechanics)
    meshes = ot.Meshes.from_files(prj.meshpaths(tmp_path))
    assert meshes.domain_name == meshes_ref.domain_name
    for name, name_ref in zip(sorted(meshes), sorted(meshes_ref), strict=True):
        assert name == name_ref
        for data in ["point_data", "cell_data", "field_data"]:
            np.testing.assert_array_equal(
                getattr(meshes[name], data).values(),
                getattr(meshes_ref[name_ref], data).values(),
            )


@pytest.mark.tools()
def test_add_from_gml(tmp_path):
    """Check, that the meshes generated from a Project + gml are correct."""
    prj = ot.Project(EXAMPLES_DIR / "prj" / "simple_mechanics.prj")
    meshes = ot.Meshes.from_files(prj.meshpaths())
    if (gml_file := prj.gml_filepath()) is not None:
        meshes.add_gml_subdomains(prj.meshpaths()[0], gml_file, tmp_path)
    subdomain_names = ["bottom", "left", "right", "top"]
    assert list(meshes.keys()) == ["square_1x1_quad_1e2"] + subdomain_names


@pytest.mark.tools()
def test_remove_material():
    """Check cells are removed drom domain and subdomains.

    Also check, that completely empty subdomains are removed entirely."""
    domain = examples.load_meshseries_THM_2D_PVD()[0]
    meshes = ot.Meshes.from_mesh(domain)

    x_mid = domain.center[0]
    dom_n_cells_right = np.count_nonzero(
        domain.cell_centers().points[:, 0] >= x_mid
    )
    assert dom_n_cells_right != domain.n_cells
    top_n_cells_right = np.count_nonzero(
        meshes["top"].cell_centers().points[:, 0] >= x_mid
    )
    left_n_cells = meshes["left"].n_cells

    x = domain.cell_centers().points[:, 0]
    domain["MaterialIDs"][(x <= x_mid)] = 99
    meshes.remove_material(99)

    assert "left" not in meshes
    assert meshes["cut_boundary"].n_cells == left_n_cells
    assert meshes["top"].n_cells == top_n_cells_right
    assert meshes["bottom"].n_cells == top_n_cells_right
    assert meshes.domain.n_cells == dom_n_cells_right
