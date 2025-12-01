"""Unit tests for meshlib."""

import textwrap

import gmsh
import pytest
import pyvista as pv

import ogstools as ot
from ogstools import examples
from ogstools.meshes._meshes_from_yaml import meshes_from_yaml


def validate_msh_file(path: str) -> list[str]:
    gmsh.initialize(["-noenv"])
    gmsh.open(path)

    gmsh.logger.start()
    messages = gmsh.logger.get()
    gmsh.logger.stop()

    gmsh.finalize()
    return messages


def test_mfy_meshes_from_yaml(tmp_path):
    yaml_content = textwrap.dedent(
        """\
    parameters: {}
    points:
        p0: { coords: [0.0, 0.0], char_length: 0.1 }
        p1: { coords: [1.0, 0.0], char_length: 0.1 }
        p2: { coords: [1.0, 1.0], char_length: 0.1 }
        p3: { coords: [0.0, 1.0], char_length: 0.1 }
    lines:
        l0: { start: p0, end: p1 }
        l1: { start: p1, end: p2 }
        l2: { start: p2, end: p3 }
        l3: { start: p3, end: p0 }
    surfaces:
        s0:
            loop: [l0, l1, l2, l3]
    groups:
        Domain:
            dim: 2
            entities: [s0]
    """
    )

    yaml_file = tmp_path / "geom.yml"
    yaml_file.write_text(yaml_content)

    msh_file = meshes_from_yaml(yaml_file, tmp_path)

    print(f"mesh-file: {msh_file}")

    # prüfen, dass .msh erzeugt wurde
    assert msh_file.exists()
    assert msh_file.suffix == ".msh"

    issues = validate_msh_file(str(msh_file))
    assert not issues, f"Mesh validation failed: {issues}"


def test_mfy_with_parameters(tmp_path):
    yaml_content = textwrap.dedent(
        """\
    parameters:
        a: 2
        b: "a * 3"
    points:
        p0: { coords: [0.0, "b"], char_length: "sqrt(4)" }
    lines: {}
    surfaces: {}
    groups: {}
    """
    )
    yaml_file = tmp_path / "geom.yml"
    yaml_file.write_text(yaml_content)
    msh_file = meshes_from_yaml(yaml_file, tmp_path)
    assert msh_file.exists()


def test_mfy_with_list_coords(tmp_path):
    yaml_content = textwrap.dedent(
        """\
    parameters: {}
    points:
        p0: { coords: ["1+1", 2], char_length: 0.1 }
    lines: {}
    surfaces: {}
    groups: {}
    """
    )
    yaml_file = tmp_path / "geom.yml"
    yaml_file.write_text(yaml_content)
    msh_file = meshes_from_yaml(yaml_file, tmp_path)
    assert msh_file.exists()


def test_mfy_invalid_type(tmp_path):
    yaml_content = textwrap.dedent(
        """\
    parameters: {}
    points:
        p0: { coords: [{foo: bar}, 0], char_length: 0.1 }
    lines: {}
    surfaces: {}
    groups: {}
    """
    )
    yaml_file = tmp_path / "geom.yml"
    yaml_file.write_text(yaml_content)

    with pytest.raises(TypeError, match="Unsupported type"):
        meshes_from_yaml(yaml_file, tmp_path)


def test_mfy_invalid_expression(tmp_path):
    yaml_content = textwrap.dedent(
        """\
    parameters:
        bad: "foo + bar"
    points:
        p0: { coords: [0, 0], char_length: "bad" }
    lines: {}
    surfaces: {}
    groups: {}
    """
    )
    yaml_file = tmp_path / "geom.yml"
    yaml_file.write_text(yaml_content)
    with pytest.raises(ValueError, match="Failed to evaluate parameter"):
        meshes_from_yaml(yaml_file, tmp_path)


def test_mfy_point_without_coords(tmp_path):
    yaml_content = textwrap.dedent(
        """\
    parameters: {}
    points:
        p0: { char_length: 0.1 }
    lines: {}
    surfaces: {}
    groups: {}
    """
    )
    yaml_file = tmp_path / "geom.yml"
    yaml_file.write_text(yaml_content)
    with pytest.raises(ValueError, match="Incomplete point definition"):
        meshes_from_yaml(yaml_file, tmp_path)


def test_mfy_surface_without_loops(tmp_path):
    yaml_content = textwrap.dedent(
        """\
    parameters: {}
    points:
        p0: { coords: [0.0, 0.0], char_length: 0.1 }
        p1: { coords: [1.0, 0.0], char_length: 0.1 }
    lines:
        l0: { start: p0, end: p1 }
    surfaces:
        s0: {}
    groups:
        Domain:
            dim: 2
            entities: [s0]
    """
    )
    yaml_file = tmp_path / "geom.yml"
    yaml_file.write_text(yaml_content)
    with pytest.raises(ValueError, match="has no 'loop' or 'loops'"):
        meshes_from_yaml(yaml_file, tmp_path)


def test_mfy_unsupported_group_dim(tmp_path):
    yaml_content = textwrap.dedent(
        """\
    parameters: {}
    points:
        p0: { coords: [0,0], char_length: 0.1 }
        p1: { coords: [1,0], char_length: 0.1 }
    lines:
        l0: { start: p0, end: p1 }
    surfaces: {}
    groups:
        TestGroup:
            dim: 3
            entities: [l0]
    """
    )
    yaml_file = tmp_path / "geom.yml"
    yaml_file.write_text(yaml_content)
    with pytest.raises(
        NotImplementedError, match="Unsupported group dimension"
    ):
        meshes_from_yaml(yaml_file, tmp_path)


def test_mfy_arc_line(tmp_path):
    yaml_content = textwrap.dedent(
        """\
    parameters: {}
    points:
        p0: { coords: [1,0], char_length: 0.1 }
        p1: { coords: [0,-1], char_length: 0.1 }
        c0: { coords: [0,0], char_length: 0.1 }
    lines:
        a0: { start: p0, end: p1, center: c0 }
    surfaces: {}
    groups: {}
    """
    )
    yaml_file = tmp_path / "geom.yml"
    yaml_file.write_text(yaml_content)
    msh_file = meshes_from_yaml(yaml_file, tmp_path)
    assert msh_file.exists()


def test_mfy_empty_blocks(tmp_path):
    yaml_content = textwrap.dedent(
        """\
    parameters: {}
    points: {}
    lines: {}
    surfaces: {}
    groups: {}
    """
    )
    yaml_file = tmp_path / "geom.yml"
    yaml_file.write_text(yaml_content)
    msh_file = meshes_from_yaml(yaml_file, tmp_path)
    assert msh_file.exists()


def test_mfy_radioactive(tmp_path):
    yaml_content = textwrap.dedent(
        """\
    parameters:
        radius_dot: 2.0
        radius_inner: 2.5
        radius_outer: 6.5
        phi_0 : 0
        phi_60 : 1.04720
        phi_120 : 2.09440
        phi_180 : 3.14159
        phi_240 : 4.18879
        phi_300 : 5.23599
        box_size : "radius_outer * 2.2"
        center : "box_size / 2"

    points:

        b00: { coords: [0.0, 0.0], char_length: 0.3 }
        b01: { coords: ["box_size", 0.0], char_length: 0.3 }
        b02: { coords: ["box_size", "box_size"], char_length: 0.3 }
        b03: { coords: [0.0, "box_size"], char_length: 0.3 }

        p00: { coords: ["center", "center"], char_length: 0.3 }
        p01: { coords: ["radius_dot+center", "center"], char_length: 0.3 }
        p02: { coords: ["center-radius_dot", "center"], char_length: 0.3 }
        p03: { coords: ["(cos(phi_0) * radius_inner)+center", "(sin(phi_0) * radius_inner)+center"], char_length: 0.3 }
        p04: { coords: ["(cos(phi_0) * radius_outer)+center", "(sin(phi_0) * radius_outer)+center"], char_length: 0.3 }
        p05: { coords: ["(cos(phi_60) * radius_outer)+center", "(sin(phi_60) * radius_outer)+center"], char_length: 0.3 }
        p06: { coords: ["(cos(phi_60) * radius_inner)+center", "(sin(phi_60) * radius_inner)+center"], char_length: 0.3 }

        p07: { coords: ["(cos(phi_120) * radius_inner)+center", "(sin(phi_120) * radius_inner)+center"], char_length: 0.3 }
        p08: { coords: ["(cos(phi_120) * radius_outer)+center", "(sin(phi_120) * radius_outer)+center"], char_length: 0.3 }
        p09: { coords: ["(cos(phi_180) * radius_outer)+center", "(sin(phi_180) * radius_outer)+center"], char_length: 0.3 }
        p10: { coords: ["(cos(phi_180) * radius_inner)+center", "(sin(phi_180) * radius_inner)+center"], char_length: 0.3 }

        p11: { coords: ["(cos(phi_240) * radius_inner)+center", "(sin(phi_240) * radius_inner)+center"], char_length: 0.3 }
        p12: { coords: ["(cos(phi_240) * radius_outer)+center", "(sin(phi_240) * radius_outer)+center"], char_length: 0.3 }
        p13: { coords: ["(cos(phi_300) * radius_outer)+center", "(sin(phi_300) * radius_outer)+center"], char_length: 0.3 }
        p14: { coords: ["(cos(phi_300) * radius_inner)+center", "(sin(phi_300) * radius_inner)+center"], char_length: 0.3 }


    lines:
        g00: { start: b00, end: b01}
        g01: { start: b01, end: b02}
        g02: { start: b02, end: b03}
        g03: { start: b03, end: b00}

        a00: { start: p01, end: p02, center: p00 }
        a01: { start: p02, end: p01, center: p00 }

        l01: { start: p03, end: p04}
        a02: { start: p04, end: p05, center: p00 }
        l02: { start: p05, end: p06}
        a03: { start: p06, end: p03, center: p00 }

        l03: { start: p07, end: p08}
        a04: { start: p08, end: p09, center: p00 }
        l04: { start: p09, end: p10}
        a05: { start: p10, end: p07, center: p00 }

        l05: { start: p11, end: p12}
        a06: { start: p12, end: p13, center: p00 }
        l06: { start: p13, end: p14}
        a07: { start: p14, end: p11, center: p00 }

    surfaces:

        s1:
            loops:
                - [a00, a01]
        s2:
            loops:
                - [l01, a02, l02, a03]
        s3:
            loops:
                - [l03, a04, l04, a05]
        s4:
            loops:
                - [l05, a06, l06, a07]

        box:
            loops:
                - [g00, g01, g02, g03]
                - ["-a00", "-a01"]
                - ["-l01", "-a02", "-l02", "-a03"]
                - ["-l03", "-a04", "-l04", "-a05"]
                - ["-l05", "-a06", "-l06", "-a07"]

    groups:
        Background:
            dim: 2
            entities: [box]
        Foreground:
            dim: 2
            entities: [s1, s2, s3, s4]
        """
    )

    yaml_file = tmp_path / "geom.yml"
    yaml_file.write_text(yaml_content)

    msh_file = meshes_from_yaml(yaml_file, tmp_path)
    print(f"mesh-file: {msh_file}")

    assert msh_file.exists()

    gmsh.initialize(["-noenv"])
    gmsh.open(str(msh_file))

    groups = gmsh.model.getPhysicalGroups()
    names = [
        gmsh.model.getPhysicalName(dim, tag).lower() for dim, tag in groups
    ]

    gmsh.finalize()

    assert "background" in names
    assert "foreground" in names
    assert len(groups) == 2

    # Check if convertible to vtk
    meshes = ot.Meshes.from_gmsh(msh_file, reindex=True, log=False)
    assert meshes, "No meshes returned"

    assert "domain" in meshes, "No 'domain' mesh found"
    assert "Background" in meshes, "No 'Background' mesh found"
    assert "Foreground" in meshes, "No 'Foreground' mesh found"

    domain = meshes["domain"]
    assert 2000 < domain.n_points < 4000, "Wrong number of points"
    assert 4000 < domain.n_cells < 8000, "Wrong number of cells"


def test_mfy_hlw_repository(tmp_path):
    # Load YAML geometry definition directly from file

    msh_file = meshes_from_yaml(examples.example_hlw, tmp_path)
    assert msh_file.exists()
    print(f"mesh-file: {msh_file}")

    # Collect physical group names as returned by gmsh
    gmsh.initialize(["-noenv"])
    gmsh.open(str(msh_file))
    groups = gmsh.model.getPhysicalGroups()
    names = [gmsh.model.getPhysicalName(dim, tag) for dim, tag in groups]
    gmsh.finalize()

    # Expected names
    expected = {
        "Observation_Point",
        "Domain_Left",
        "Domain_Top",
        "Domain_Right",
        "Domain_Bottom",
        "Hostrock",
        "EDZ",
        "Support",
        "Floor",
        "Canister",
        "Filling",
    }

    missing = expected - set(names)
    unexpected = set(names) - expected
    assert not missing, f"Missing expected groups: {missing}"
    assert not unexpected, f"Unexpected extra groups: {unexpected}"

    # Check if convertible to vtk
    meshes = ot.Meshes.from_gmsh(msh_file, reindex=True, log=False)
    assert meshes, "No meshes returned"

    # Build expected mesh keys: "domain" + prefixed group names
    expected_meshes = {"domain"} | {f"{name}" for name in expected}

    missing_meshes = expected_meshes - set(meshes.keys())
    unexpected_meshes = set(meshes.keys()) - expected_meshes
    assert not missing_meshes, f"Missing expected meshes: {missing_meshes}"
    assert (
        not unexpected_meshes
    ), f"Unexpected extra meshes: {unexpected_meshes}"

    domain = meshes["domain"]
    assert 3000 < domain.n_points < 4000, "Wrong number of points"
    assert 5000 < domain.n_cells < 7000, "Wrong number of cells"


def test_mfy_hlw_repository_meshes_container():
    meshes = ot.Meshes.from_yaml(examples.example_hlw)

    # Check if the keys match
    expected = {
        "Observation_Point",
        "Domain_Left",
        "Domain_Top",
        "Domain_Right",
        "Domain_Bottom",
        "Hostrock",
        "EDZ",
        "Support",
        "Floor",
        "Canister",
        "Filling",
    }

    expected_meshes = {"domain"} | {f"{name}" for name in expected}

    assert set(meshes.keys()) == expected_meshes, (
        f"Unexpected mesh keys: {set(meshes.keys()) - expected_meshes} "
        f"Missing: {expected_meshes - set(meshes.keys())}"
    )

    # Check domain
    domain = meshes.domain
    assert domain.n_points > 2000
    assert domain.n_cells > 4000

    # Test access to subdomains
    subdomains = meshes.subdomains
    assert all(isinstance(m, pv.UnstructuredGrid) for m in subdomains.values())
    assert "Floor" in subdomains
    assert "Canister" in subdomains

    # Test saving (writes temporary VTUs)
    files = meshes.save()
    assert all(f.exists() for f in files)
