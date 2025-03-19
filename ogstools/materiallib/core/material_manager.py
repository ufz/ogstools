# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import]

import ogstools.definitions as defs
from ogstools.materiallib.schema.process_schema import PROCESS_SCHEMAS

from .material import Material

logger = logging.getLogger(__name__)


class MaterialManager:
    """
    Manager for material definitions loaded from a repository of YAML files.

    A MaterialManager can be constructed in two ways:

    1. **From a repository directory** (default):
       If no materials are provided, all YAML files in the given
       `data_dir` are parsed into Material objects and stored in
       `materials_db`.

    2. **From pre-filtered materials**:
       If a dictionary of `Material` objects is passed via `materials`,
       the manager is created directly from these. This is typically
       used internally when creating filtered views.

    Once constructed, a MaterialManager can:
    - provide access to the stored materials

    - filter materials according to a process schema, subdomain mapping,
      and/or fluid assignments

    - represent the filtered material set as a new MaterialManager.

    """

    def __init__(
        self,
        data_dir: Path | None = None,
        materials: dict[str, Material] | None = None,
        subdomain_ids: dict[str, int] | None = None,
        process: str | None = None,
    ):
        """
        Initialize a MaterialManager instance.

        Parameters
        ----------
        data_dir : Path | None
            Directory containing the repository of material YAML files.
            Defaults to `defs.MATERIALS_DIR`. Only used if no `materials`
            are passed.
        materials : dict[str, Material] | None
            Pre-loaded material dictionary. If None, materials are loaded
            from the repository directory. If provided, these materials are
            used as-is without accessing the repository (commonly from filtering).
        subdomain_ids : dict[str, int] | None
            Mapping of subdomain names to material IDs.
        process : str | None
            Process type used for filtering, if applicable.

        Notes
        -----
        - If `materials` is None, the instance represents the **full material
          repository** loaded from the given directory.
        - If `materials` is provided, the instance represents a **filtered view**
          and does not perform any additional repository access.
        """
        self.data_dir = data_dir or Path(defs.MATERIALS_DIR)
        self.materials_db: dict[str, Material] = materials or {}
        self.subdomain_ids: dict[str, int] = subdomain_ids or {}
        self.fluids: dict[str, Material] = {}
        self.process = process

        if materials is None:  # only load from repository if not provided
            logger.info("Loading materials from repository: %s", self.data_dir)
            self._load_materials()
            logger.debug("Materials loaded: %s", list(self.materials_db.keys()))
        else:
            logger.debug("Using provided materials: %s", list(materials.keys()))

    # ------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------
    def _load_materials(self) -> None:
        yaml_files = list(self.data_dir.glob("*.yml")) + list(
            self.data_dir.glob("*.yaml")
        )
        if not yaml_files:
            msg = f"No YAML files found in {self.data_dir}"
            raise FileNotFoundError(msg)

        for file_path in yaml_files:
            with file_path.open(encoding="utf-8") as file:
                raw_data = yaml.safe_load(file)

                if not isinstance(raw_data, dict):
                    logger.debug("Skipping invalid YAML file: %s", file_path)
                    continue

                if "name" not in raw_data:
                    logger.debug(
                        "Skipping YAML file without 'name': %s", file_path
                    )
                    continue

                name = raw_data["name"]
                material = Material(name=name, raw_data=raw_data)
                self.materials_db[material.name] = material

    # ------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------
    def get_material(self, name: str) -> Material | None:
        """
        Retrieve a material from the repository by name.
        """
        return self.materials_db.get(name)

    def _list_materials(self) -> list[str]:
        return list(self.materials_db.keys())

    # ------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------
    def filter(
        self,
        process: str,
        subdomains: list[dict[str, Any]],
        fluids: dict[str, str] | None = None,
    ) -> MaterialManager:
        """
        Create a filtered view of the materials for a given process.

        Filtering is based on:
        - a process schema,
        - subdomain assignments (mapping subdomain names to one or more material IDs),
        - optional fluid phase materials.

        Parameters
        ----------
        process : str
            The process name to filter by.
        subdomains : list[dict[str, Any]]
            A list of dictionaries, each containing the material name,
            a subdomain name, and one or more material IDs.
        fluids : dict[str, str] | None
            A dictionary mapping phase types to material names.

        Returns
        -------
        MaterialManager
            A new MaterialManager instance containing only the filtered
            materials. This new instance does not access the repository;
            it reuses the already loaded Material objects.
        """
        schema = PROCESS_SCHEMAS.get(process)
        if schema is None:
            msg = f"No process schema found for '{process}'"
            raise ValueError(msg)

        filtered: dict[str, Material] = {}
        subdomain_ids: dict[str, int] = {}

        # Solids (subdomains)
        for entry in subdomains:
            name = entry["material"]
            mat = self.get_material(name)
            if mat is None:
                msg = f"Material '{name}' not found in repository."
                raise ValueError(msg)

            filtered_mat = mat.filter_process(schema)

            subdomain_name = entry["subdomain"]
            mat_ids = entry["material_ids"]
            if isinstance(mat_ids, int):
                mat_ids = [mat_ids]

            for mat_id in mat_ids:
                # if only one ID, keep the plain subdomain name
                # if multiple IDs, disambiguate by appending the ID
                key = (
                    subdomain_name
                    if len(mat_ids) == 1
                    else f"{subdomain_name}_{mat_id}"
                )
                filtered[key] = filtered_mat
                subdomain_ids[key] = mat_id

        # Fluids
        fluid_materials: dict[str, Material] = {}
        for phase_type, mat_name in (fluids or {}).items():
            raw = self.get_material(mat_name)
            if raw is None:
                msg = f"Fluid material '{mat_name}' not found in repository."
                raise ValueError(msg)

            fluid_materials[phase_type] = raw.filter_process(schema)

        return MaterialManager(
            data_dir=self.data_dir,
            materials=filtered,
            subdomain_ids=subdomain_ids,
            process=process,
        )._with_fluids(fluid_materials)

    def _with_fluids(self, fluids: dict[str, Material]) -> MaterialManager:
        """
        Internal helper to attach fluid materials to a filtered manager.
        """
        self.fluids = fluids
        return self

    def _list_ids(self) -> list[int]:
        """
        Return a list of material IDs for the subdomains.
        """
        return list(self.subdomain_ids.values())

    def _list_subdomains(self) -> list[str]:
        """
        Return a list of subdomain names managed by this instance.
        """
        return list(self.subdomain_ids.keys())

    # ------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------
    def __repr__(self) -> str:
        lines = [f"<MaterialManager with {len(self.materials_db)} materials>"]

        # Solids / medium materials
        if self.subdomain_ids:
            lines.append(
                f"  ├─ {len(self.subdomain_ids)} solid material entries mapped to material_ids:"
            )
            for name in sorted(
                self.subdomain_ids, key=lambda n: self.subdomain_ids.get(n, 999)
            ):
                mat = self.materials_db.get(name)
                mid = self.subdomain_ids.get(name, "?")
                lines.append(
                    f"  │   [{mid}] {name}: '{mat.name if mat else '?'}'"
                )
        else:
            lines.append("  ├─ No solid or medium materials defined")

        # Fluids: prefer explicit self.fluids if present
        if self.fluids:
            lines.append(f"  ├─ {len(self.fluids)} fluid materials:")
            for phase_type, mat in self.fluids.items():
                lines.append(f"  │   {phase_type}: {mat.name}")
        else:
            # fallback: derive fluids from materials_db
            fluid_keys = [
                k for k in self.materials_db if k not in self.subdomain_ids
            ]
            if fluid_keys:
                lines.append(f"  ├─ {len(fluid_keys)} fluid materials:")
                for phase_type in fluid_keys:
                    mat = self.materials_db[phase_type]
                    lines.append(f"  │   {phase_type}: {mat.name}")
            else:
                lines.append("  └─ No fluid materials assigned")

        return "\n".join(lines)
