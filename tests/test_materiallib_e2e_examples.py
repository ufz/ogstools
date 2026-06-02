# SPDX-FileCopyrightText: Copyright (c) OpenGeoSys Community (opengeosys.org)
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml  # type: ignore[import]

import ogstools as ot
from ogstools import Project, examples
from ogstools.materiallib.core.material_manager import MaterialManager
from ogstools.materiallib.core.media import MediaSet


@pytest.mark.system
def test_heat_conduction_project_runs_with_grouped_domain_materials(
    tmp_path: Path,
) -> None:
    project = Project(input_file=examples.prj_heat_conduction).copy()

    rect = ot.gmsh_tools.rect(
        lengths=(1, 1),
        n_edge_cells=(2, 2),
        n_layers=1,
        structured_grid=True,
        order=1,
        jiggle=0.0,
    )
    meshes = ot.Meshes.from_gmsh(rect, log=False)
    meshes["domain"].cell_data["MaterialIDs"] = np.full(
        meshes["domain"].n_cells, 4, dtype=np.int32
    )

    db = MaterialManager()
    filtered = db.filter(
        process="T",
        subdomains=[
            {
                "subdomain": "region",
                "material": "opalinus_clay",
                "material_ids": [4],
            }
        ],
        fluids={},
    )

    project.set_media(MediaSet(filtered))
    model = ot.Model(project, meshes)
    simulation = model.run(target=tmp_path / "heat_conduction_run")

    assert simulation.status == ot.Simulation.Status.done

    text = project.prjfile.read_text(encoding="utf-8")
    assert "<media>" in text
    assert "thermal_conductivity" in text
    assert "specific_heat_capacity" in text
    assert "density" in text
    assert "opalinus_clay" not in text


@pytest.mark.system
def test_th2m_pt_example_project_runs_with_grouped_domain_materials(
    tmp_path: Path,
) -> None:
    source_project_dir = (
        Path(__file__).resolve().parents[1]
        / "ogstools/examples/prj/TH2M/TH2M_PT"
    )
    project_dir = tmp_path / "TH2M_PT"
    shutil.copytree(source_project_dir, project_dir)

    source_material_dir = (
        Path(__file__).resolve().parents[1] / "ogstools/examples/materiallib"
    )
    material_dir = project_dir / "materials"
    shutil.copytree(source_material_dir, material_dir)

    diffusion_path = material_dir / "diffusion_coefficients.yml"
    diffusion_data = yaml.safe_load(diffusion_path.read_text(encoding="utf-8"))
    diffusion_data.setdefault("Gas", {}).setdefault("hydrogen", {})[
        "water"
    ] = 1.0e-06
    diffusion_data.setdefault("AqueousLiquid", {}).setdefault("water", {})[
        "hydrogen"
    ] = 2.1e-09
    diffusion_path.write_text(
        yaml.safe_dump(diffusion_data, sort_keys=False), encoding="utf-8"
    )

    project = Project(
        input_file=project_dir / "th2m_phase_transition.prj"
    ).copy()

    db = MaterialManager(data_dir=material_dir)
    filtered = db.filter(
        process="TH2M_PT",
        subdomains=[
            {
                "subdomain": "host_rock",
                "material": "opalinus_clay",
                "material_ids": [0],
            }
        ],
        fluids={"AqueousLiquid": "water", "Gas": "hydrogen"},
    )

    project.set_media(MediaSet(filtered))
    model = ot.Model(project)
    simulation = model.run(target=project_dir / "run")

    assert simulation.status == ot.Simulation.Status.done

    text = project.prjfile.read_text(encoding="utf-8")
    assert "<media>" in text
    assert "AqueousLiquid" in text
    assert "Gas" in text
    assert "opalinus_clay" not in text
    assert "hydrogen" in text
