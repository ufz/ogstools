# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

import copy
import inspect
from pathlib import Path
from typing import Any

import yaml
from typing_extensions import Self

from .storage import StorageBase


class Execution(StorageBase):
    """
    Configuration for OGS simulation execution parameters.

    This class encapsulates all settings related to how an OGS simulation
    is executed, including parallelization, containerization, and logging options.
    """

    def __init__(
        self,
        interactive: bool = False,
        args: Any | None = None,
        container_path: Path | str | None = None,
        wrapper: Any | None = None,
        mpi_ranks: int | None = None,
        ogs_bin_path: Path | str = Path("ogs"),
        omp_num_threads: int | None = None,
        ogs_asm_threads: int | None = None,
        write_logs: bool = True,
        log_level: str = "info",
        id: str | None = None,
    ) -> None:
        """
        Initialize an Execution configuration.

        :param interactive:     If True, use interactive mode for stepwise control.
        :param args:            Additional command-line arguments for OGS.
        :param container_path:  Path to a container (e.g., Docker, Singularity).
        :param wrapper:         Custom wrapper command for execution.
        :param mpi_ranks:       Number of MPI ranks for parallel execution.
        :param ogs_bin_path:    Path to the OGS binary. Defaults to "ogs".
        :param omp_num_threads: Number of OpenMP threads per MPI rank.
        :param ogs_asm_threads: Number of assembly threads for OGS.
        :param write_logs:      If True, write log output to file.
        :param log_level:       Logging level (e.g., "info", "debug", "warn").
        :param id:              Optional unique identifier for this execution config.
        """
        super().__init__("Execution", file_ext="yaml", id=id)
        self.interactive = interactive
        self.container_path = container_path
        self.wrapper = wrapper
        self.args = args
        self.mpi_ranks = mpi_ranks
        self.ogs_bin_path = ogs_bin_path
        self.omp_num_threads = omp_num_threads
        self.ogs_asm_threads = ogs_asm_threads
        self.write_logs = write_logs
        self.log_level = log_level

    @classmethod
    def from_file(cls, filepath: Path | str) -> Self:
        """
        Restore an Execution object from an execution.yaml file.

        :param filepath:    Path to execution.yaml file.
        :returns:           Restored Execution instance.

        :raises FileNotFoundError: If the specified file does not exist.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            msg = f"Execution file does not exist: {filepath}"
            raise FileNotFoundError(msg)

        with filepath.open("r") as f:
            data = yaml.safe_load(f) or {}

        if "ogs_bin_path" in data:
            data["ogs_bin_path"] = Path(data["ogs_bin_path"])

        execution = cls(**data)
        execution._bind_to_path(filepath)
        return execution

    @classmethod
    def from_id(cls, execution_id: str) -> "Execution":
        """
        Load Execution from the user storage path using its ID.
        StorageBase.Userpath must be set.

        :param execution_id: The unique ID of the Execution to load.
        :returns:            An Execution instance restored from disk.
        """
        execution_file = (
            StorageBase.saving_path()
            / "Execution"
            / f"{execution_id}"
            / "meta.yaml"
        )

        if not execution_file.exists():
            msg = f"No execution file found at {execution_file}"
            raise FileNotFoundError(msg)

        execution = cls.from_file(execution_file)
        execution._id = execution_id
        return execution

    def _save_impl(self, dry_run: bool = False) -> list[Path]:
        target = Path(self.next_target)
        if dry_run:
            return [target]

        target.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "interactive": self.interactive,
            "args": self.args,
            "container_path": (
                str(self.container_path)
                if self.container_path is not None
                else None
            ),
            "wrapper": self.wrapper,
            "mpi_ranks": self.mpi_ranks,
            "ogs_bin_path": str(self.ogs_bin_path),
            "omp_num_threads": self.omp_num_threads,
            "ogs_asm_threads": self.ogs_asm_threads,
            "write_logs": self.write_logs,
            "log_level": self.log_level,
        }

        with target.open("w") as f:
            yaml.safe_dump(data, f, sort_keys=False)

        return [target]

    def save(
        self,
        target: Path | str | None = None,
        overwrite: bool | None = None,
        dry_run: bool = False,
        archive: bool = False,
        id: str | None = None,
    ) -> list[Path]:

        user_defined = self._pre_save(target, overwrite, dry_run, id=id)
        files = self._save_impl(dry_run=dry_run)
        self._post_save(user_defined, archive, dry_run)
        return files

    def _propagate_target(self) -> None:
        pass

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        attrs = self._non_default_attributes()
        base_repr = super().__repr__()

        if self.user_specified_id:
            construct = f'{cls_name}.from_id("{self._id}")'
        elif self.is_saved:
            construct = f"{cls_name}.from_file({str(self.active_target)!r})"
        else:
            attrs_filtered = {k: v for k, v in attrs.items() if k != "id"}
            inner = ", ".join(f"{k}={v!r}" for k, v in attrs_filtered.items())
            construct = f"{cls_name}({inner})"

        return f"{construct}\n{base_repr}"

    def __str__(self) -> str:
        base_str = super().__str__()
        non_defaults = self._non_default_attributes()
        lines = [base_str, "  Execution settings:"]
        signature = inspect.signature(self.__class__.__init__)
        for name, param in signature.parameters.items():
            if name == "self" or param.default is inspect._empty:
                continue
            value = getattr(self, name)
            marker = "*" if name in non_defaults else " "
            lines.append(f"    {marker} {name}: {value}")
        return "\n".join(lines)

    def _value_attrs(self) -> dict[str, object]:
        """Return attributes that define value (excludes StorageBase state)."""
        return {
            k: v
            for k, v in self.__dict__.items()
            if k not in self._SAVE_STATE_ATTRS
        }

    def __deepcopy__(self, memo: dict) -> "Execution":
        if id(self) in memo:
            return memo[id(self)]

        new = Execution()
        for k, v in self._value_attrs().items():
            setattr(new, k, copy.deepcopy(v, memo))
        memo[id(self)] = new
        return new

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Execution):
            return NotImplemented

        return self._value_attrs() == other._value_attrs()
