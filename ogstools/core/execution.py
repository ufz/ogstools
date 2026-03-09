# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

import copy
import inspect
import os
import shutil
import subprocess
import sys
import warnings
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

    __hash__ = None
    _container_exec: str = "apptainer exec"
    mpi_wrapper: str = "mpirun -np"
    ogs_serial: str = (
        "https://vip.s3.ufz.de/ogs/public/container/ogs/master/ogs-serial.squashfs ogs"
    )
    ogs_parallel: str = (
        "https://vip.s3.ufz.de/ogs/public/container/ogs/master/ogs-petsc.squashfs ogs"
    )

    def __init__(
        self,
        interactive: bool = False,
        args: Any | None = None,
        mpi_wrapper: str = "mpirun -np",
        wrapper: str | None = None,
        ogs_serial: str = "ogs",
        ogs_parallel: str = ogs_parallel,
        mpi_ranks: int | None = None,
        omp_num_threads: int | None = None,
        ogs_asm_threads: int | None = None,
        write_logs: bool = True,
        log_level: str = "info",
        id: str | None = None,
    ) -> None:
        """
        Initialize an Execution configuration.

        :param interactive:  If True, use interactive mode for stepwise control.
        :param args:         Additional command-line arguments for OGS.
        :param wrapper:      Generic command prefix prepended before the full command.
                             E.g. ``"valgrind"``, ``"perf stat"``, ``"numactl --cpunodebind=0"``.
        :param mpi_wrapper:  MPI launcher prefix (e.g. "mpirun -np", "srun --ntasks"). Defaults to "mpirun -np".
        :param ogs_serial:   How to run OGS for sequential (non-MPI) execution.
                             Either a binary path (``"ogs"``, ``"/usr/bin/ogs"``) or a
                             container URL / file followed by the binary name inside the
                             container (``"https://...serial.squashfs ogs"``).
        :param ogs_parallel: How to run OGS for parallel (MPI) execution.
                             Same format as ``ogs_serial``; selected when ``mpi_ranks`` is set.
        :param mpi_ranks:       Number of MPI ranks for parallel execution.
        :param omp_num_threads: Number of OpenMP threads per MPI rank.
        :param ogs_asm_threads: Number of assembly threads for OGS.
        :param write_logs:      If True, write log output to file.
        :param log_level:       Logging level (e.g., "info", "debug", "warn").
        :param id:              Optional unique identifier for this execution config.

        .. note::
            ``mpi_wrapper`` is a class attribute (default ``"mpirun -np"``) and can be
            overridden per instance: ``exe.mpi_wrapper = "srun --ntasks"``.
        """
        super().__init__("Execution", file_ext="yaml", id=id)
        self.env: dict[str, str] = os.environ.copy()
        self.interactive = interactive
        self.wrapper = wrapper
        self.mpi_wrapper = mpi_wrapper
        self.ogs_serial = ogs_serial
        self.ogs_parallel = ogs_parallel
        self.args = args
        self.mpi_ranks = mpi_ranks
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

        self._validate_container()
        self._validate_ogs()
        if dry_run:
            return [target]

        target.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "interactive": self.interactive,
            "args": self.args,
            "wrapper": self.wrapper,
            "mpi_wrapper": self.mpi_wrapper,
            "ogs_serial": self.ogs_serial,
            "ogs_parallel": self.ogs_parallel,
            "mpi_ranks": self.mpi_ranks,
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

    def __setattr__(self, name: str, value: object) -> None:
        super().__setattr__(name, value)
        if name == "omp_num_threads":
            if value is None:
                self.env.pop("OMP_NUM_THREADS", None)
            else:
                self.env["OMP_NUM_THREADS"] = str(value)
        elif name == "ogs_asm_threads":
            if value is None:
                self.env.pop("OGS_ASM_THREADS", None)
            else:
                self.env["OGS_ASM_THREADS"] = str(value)
        if name in ("mpi_ranks", "omp_num_threads"):
            mpi = self.__dict__.get("mpi_ranks") or 1
            cpu_count = os.cpu_count() or 1
            omp = self.__dict__.get("omp_num_threads") or cpu_count
            if mpi * omp > cpu_count:
                warnings.warn(
                    f"omp_num_threads={omp} and mpi_ranks={mpi}: "
                    f"total threads ({mpi * omp}) exceeds system CPU count ({cpu_count}), "
                    "which may cause resource over-subscription.",
                    UserWarning,
                    stacklevel=2,
                )

    def _value_attrs(self) -> dict[str, object]:
        """Return attributes that define value (excludes StorageBase state and env)."""
        return {
            k: v
            for k, v in self.__dict__.items()
            if k not in self._SAVE_STATE_ATTRS and k != "env"
        }

    def __deepcopy__(self, memo: dict) -> "Execution":
        if id(self) in memo:
            return memo[id(self)]

        new = Execution()
        for k, v in self._value_attrs().items():
            setattr(new, k, copy.deepcopy(v, memo))
        new.env = os.environ.copy()
        if new.omp_num_threads is not None:
            new.env["OMP_NUM_THREADS"] = str(new.omp_num_threads)
        if new.ogs_asm_threads is not None:
            new.env["OGS_ASM_THREADS"] = str(new.ogs_asm_threads)
        memo[id(self)] = new
        return new

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Execution):
            return NotImplemented

        return self._value_attrs() == other._value_attrs()

    @property
    def container_prefix(self) -> str | None:
        """Container launch prefix, e.g. 'apptainer exec /path/to.sif', or None."""
        if self.container_path is None:
            return None
        is_url = str(self.container_path).startswith(("http://", "https://"))
        ref = (
            str(self.container_path)
            if is_url
            else str(Path(self.container_path).expanduser())
        )
        return f"{self._container_exec} {ref}"

    def _validate_container(self) -> None:
        """Validate container path and launcher availability."""
        if self.container_path is None:
            return
        container_str = str(self.container_path)
        if Path(container_str).suffix.lower() not in (".sif", ".squashfs"):
            msg = "Container must be a *.sif or *.squashfs file."
            raise RuntimeError(msg)
        if sys.platform == "win32":
            msg = "Running OGS in a container is only possible on Linux."
            raise RuntimeError(msg)
        launcher = self._container_exec.split()[0]
        if shutil.which(launcher) is None:
            msg = f"Container launcher '{launcher}' was not found."
            raise RuntimeError(msg)
        is_url = container_str.startswith(("http://", "https://"))
        if not is_url:
            container = Path(self.container_path).expanduser()
            if not container.is_file():
                msg = "Container path is not a file."
                raise RuntimeError(msg)

    @staticmethod
    def _parse_ogs(ogs_cmd: str) -> tuple[str | None, str]:
        """Parse an OGS run-spec into ``(container_ref, binary)``.

        A run-spec is either a plain binary path (``"ogs"``, ``"/usr/bin/ogs"``) or
        a container reference followed by the binary name inside the container
        (``"https://...serial.squashfs ogs"``, ``"/path/to/local.sif ogs"``).

        The first whitespace-delimited token is treated as a container reference
        when it is a URL (``http://`` / ``https://``) or ends with ``.sif`` /
        ``.squashfs``.  *container_ref* is ``None`` for native execution.
        """
        parts = ogs_cmd.split(None, 1)
        first = parts[0]
        binary = parts[1] if len(parts) > 1 else "ogs"
        is_url = first.startswith(("http://", "https://"))
        is_container_file = Path(first).suffix.lower() in (".sif", ".squashfs")
        if is_url or is_container_file:
            return first, binary
        return None, ogs_cmd.strip()

    @property
    def _active_ogs(self) -> str:
        """Active OGS run-spec: ``ogs_parallel`` when ``mpi_ranks`` is set, else ``ogs_serial``."""
        return (
            self.ogs_parallel if self.mpi_ranks is not None else self.ogs_serial
        )

    @property
    def container_path(self) -> str | None:
        """Container reference for the active run, or ``None`` for native execution."""
        container, _ = self._parse_ogs(self._active_ogs)
        return container

    @property
    def ogs_resolved_path(self) -> str:
        """Full path to the OGS executable.

        For containerized execution the binary name is returned unchanged
        (resolution happens inside the container).  For native execution the
        path is resolved via :func:`shutil.which`.
        """
        container, binary = self._parse_ogs(self._active_ogs)
        if container is not None:
            return binary
        resolved = shutil.which(binary)
        return resolved if resolved is not None else binary

    def _validate_ogs(self) -> None:
        """Validate the OGS executable and check PETSc configuration.

        For native execution the executable must be locatable via PATH.
        Running ``ogs --version`` must succeed.
        For parallel execution (``mpi_ranks`` is set) the version output must
        contain ``-DOGS_USE_PETSC="ON"``.
        For sequential execution ``-DOGS_USE_PETSC="ON"`` must be absent.
        """
        parallel = self.mpi_ranks is not None

        if self.container_path is not None:
            version_cmd = (
                f"{self.container_prefix} {self.ogs_resolved_path} --version"
            )
        else:
            _, binary = self._parse_ogs(self._active_ogs)
            resolved = shutil.which(binary)
            if resolved is None:
                msg = (
                    f"OGS executable '{binary}' was not found. "
                    "See https://www.opengeosys.org/docs/userguide/basics/introduction/ "
                    "for installation instructions."
                )
                raise RuntimeError(msg)
            version_cmd = f"{resolved} --version"

        result = subprocess.run(
            version_cmd,
            shell=True,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            msg = (
                f"OGS executable failed to run '--version' "
                f"(exit code {result.returncode}): {result.stderr or result.stdout}"
            )
            raise RuntimeError(msg)

        has_petsc = '-DOGS_USE_PETSC="ON"' in result.stdout

        if parallel and not has_petsc:
            msg = (
                "Parallel execution requires OGS built with PETSc "
                '(-DOGS_USE_PETSC="ON"), but it was not found in '
                "'ogs --version' output."
            )
            raise RuntimeError(msg)
        if not parallel and has_petsc:
            msg = (
                "Sequential execution requires OGS built without PETSc "
                '(-DOGS_USE_PETSC must be absent or "OFF"), '
                "but 'ogs --version' reports -DOGS_USE_PETSC=\"ON\". "
                "Use a non-PETSc OGS build for sequential execution."
            )
            raise RuntimeError(msg)
