# SPDX-FileCopyrightText: Copyright (c) OpenGeoSys Community (opengeosys.org)
# SPDX-License-Identifier: BSD-3-Clause

import json
import logging
from collections.abc import Iterator, MutableMapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyvista as pv

logging.basicConfig()  # Important, initializes root logger
logger = logging.getLogger(__name__)


@dataclass
class IPdict:
    order: int
    num_components: int
    values: np.ndarray


def ip_metadata(
    mesh: pv.UnstructuredGrid,
) -> dict[str, list[dict[str, Any]]] | None:
    "return the IntegrationPointMetaData in the mesh's field_data as a dict."
    if "IntegrationPointMetaData" not in mesh.field_data:
        return None
    data = bytes(mesh.field_data["IntegrationPointMetaData"]).decode("utf-8")
    return json.loads(data)


def _set_ip_metadata(
    mesh: pv.UnstructuredGrid, ip_metadata: dict[str, list[dict[str, Any]]]
) -> None:
    "Write IntegrationPointMetaData into the mesh's field_data."
    ip_meta_str = json.dumps(ip_metadata)
    mesh.field_data["IntegrationPointMetaData"] = np.frombuffer(
        ip_meta_str.encode("utf-8"), dtype=np.int8
    )


class IPdata(MutableMapping):
    """An interface to the integration point data of a mesh."""

    def __init__(
        self, mesh: pv.UnstructuredGrid, auto_sync: bool = True
    ) -> None:
        """
        Initialize an IPdata object

        :param mesh: Mesh, which should contain integration point metadata
        :param auto_sync: By default, automatically syncs changes in this
            object with the given mesh. If set to False, `sync` should be
            manually called after all changes to IPdata.
        """
        self.mesh = mesh
        ip_meta = ip_metadata(mesh)
        self._array_map: dict[str, IPdict]
        self.auto_sync = auto_sync
        if ip_meta is None:
            self._array_map = {}
        else:
            self._array_map = {
                kwargs["name"]: IPdict(
                    order=kwargs["integration_order"],
                    num_components=kwargs["number_of_components"],
                    values=mesh.field_data[kwargs["name"]],
                )
                for kwargs in ip_meta["integration_point_arrays"]
            }

    def __setitem__(self, key: str, value: IPdict) -> None:
        self._array_map[key] = value
        if self.auto_sync:
            self._sync()

    def __getitem__(self, key: str) -> IPdict:
        return self._array_map[key]

    def __delitem__(self, key: str) -> None:
        del self._array_map[key]
        self.mesh.field_data.remove(key)
        if self.auto_sync:
            self._sync()

    def __iter__(self) -> Iterator[str]:
        return self._array_map.__iter__()

    def __len__(self) -> int:
        return len(self._array_map)

    def __contains__(self, key: object) -> bool:
        return key in self._array_map

    def __str__(self) -> str:
        lines = [
            f"{name} (order={data.order}, "
            f"num_components={data.num_components}, "
            f"len={len(data.values)}"
            for name, data in self.items()
        ]
        return "\n".join(lines)

    @property
    def info(self) -> str:
        return (
            str(self._array_map)
            .replace("values=", "values=\n")
            .replace("))),", "))),\n")
        )

    @property
    def n_points(self) -> int:
        """Read number of integration points

        return 0, if no integartion point metadata is present in the mesh."""
        ip_meta = ip_metadata(self.mesh)
        if ip_meta is None:
            logger.info(
                "No integration point metadata present. "
                "Can't infer number of integration points"
            )
            return 0
        from .ip_mesh import to_ip_point_cloud

        return to_ip_point_cloud(self.mesh).number_of_points

    def set(
        self,
        name: str,
        order: int,
        num_components: int,
        values: np.ndarray | list[float | int] | float | int,
    ) -> None:
        """set integration point data according to the given values."""
        if isinstance(values, int | float | list):
            n_pts = self.n_points
            if n_pts == 0:
                msg = "No ip metadata present. Cannot broadcast scalars."
                raise TypeError(msg)
            vals_ = np.full((n_pts, num_components), values)
        else:
            vals_ = values
        self[name] = IPdict(
            order=order, num_components=num_components, values=vals_
        )

    def _sync(self) -> None:
        """Sync the current integration point data with the mesh.

        This updates the integration point metadata as well as the corresponding
        values in the Mesh's field_data.
        """
        if len(self) == 0:
            return
        ip_meta = {
            "integration_point_arrays": [
                {
                    "integration_order": array.order,
                    "name": name,
                    "number_of_components": array.num_components,
                }
                for name, array in self._array_map.items()
            ]
        }
        _set_ip_metadata(self.mesh, ip_meta)
        for name, array in self._array_map.items():
            self.mesh.field_data[name] = array.values
