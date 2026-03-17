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

import yaml
from typing_extensions import Self

from .storage import StorageBase


class Execution(StorageBase):
    """
    Configuration for OGS simulation execution parameters.

    This class encapsulates all settings related to how an OGS simulation
    is executed, including parallelization, containerization, and logging options.

    **OGS binary / container selection**

    The ``ogs`` parameter accepts either:

    - A directory containing the OGS executable (``"/path/to/bin"``).
    - A container image path or URL (``.sif`` / ``.squashfs``), in which case
      OGS is run inside the container as ``ogs``.

    Use pre-built container image URLs::

        Execution(ogs=Execution.CONTAINER_PARALLEL_V6_5_7)

    When ``ogs`` is not set, the executable is looked up via ``OGS_BIN_PATH``
    and, if not defined, on PATH.

    **Site-wide defaults via** ``OGS_EXECUTION_DEFAULTS``

    Point the environment variable to a YAML file (e.g. in ``.envrc``, ``activate``,
    or ``.bashrc``) to set site-wide defaults::

        export OGS_EXECUTION_DEFAULTS=/path/to/my_execution_defaults.yaml

    A typical cluster defaults file::

        ogs:         "/scratch/containers/ogs-6.5.7-petsc.squashfs"
        mpi_wrapper: "srun --ntasks"
        write_logs:  true

    A typical developer defaults file::

        ogs: "ogs/build/release/bin"

    """

    __hash__ = None
    _CONTAINER_EXEC: str = "apptainer exec"
    default: "Execution"
    CONTAINER_SERIAL = "https://vip.s3.ufz.de/ogs/public/binaries/ogs6/6.5.7/ogs-6.5.7-serial.squashfs"
    CONTAINER_PARALLEL = "https://vip.s3.ufz.de/ogs/public/binaries/ogs6/6.5.7/ogs-6.5.7-petsc.squashfs"

    """Default :class:`Execution` instance, created at import time via auto-detection.

    Use this to inspect or share the default execution configuration without
    creating a new object each time.

    Users can override the default for the entire session::

        Execution.default = Execution(ogs="/path/to/bin")
    """

    def __init__(
        self,
        interactive: bool = False,
        args: str | None = None,
        mpi_wrapper: str | None = "mpirun -np",
        wrapper: str | None = None,
        ogs: str | None = None,
        mpi_ranks: int | None = None,
        omp_num_threads: int | None = None,
        ogs_asm_threads: int | None = None,
        write_logs: bool = True,
        log_level: str | None = None,
        id: str | None = None,
    ) -> None:
        """
        Initialize an Execution configuration.

        :param interactive:  If True, use interactive mode for stepwise control.
        :param args:         Extra OGS command-line flags appended verbatim
                             (see ``ogs --help`` for the full list).
                             Useful flags:
                             Example: ``"--write-prj --log-parallel"``.
        :param mpi_wrapper:  MPI launcher prefix,
                             e.g. ``"mpirun -np"``, ``"mpiexec -n"``, ``"srun --ntasks"``.
        :param wrapper:      Generic command prefix prepended before the full command,
                             e.g. ``"valgrind"``, ``"perf stat"``.
        :param ogs:          Directory containing the OGS executable, or a container
                             image path/URL (``.sif`` / ``.squashfs``).
                             When not set, looked up via ``OGS_BIN_PATH`` or PATH.
        :param mpi_ranks:       Number of MPI ranks. ``None`` = serial run.
        :param omp_num_threads: OpenMP threads per MPI rank. ``None`` = let OGS decide.
                                See `OpenMP parallelization <https://www.opengeosys.org/6.5.7/docs/userguide/basics/openmp/>`_.
        :param ogs_asm_threads: OGS assembly threads. ``None`` = let OGS decide.
                                See `OpenMP parallelization <https://www.opengeosys.org/6.5.7/docs/userguide/basics/openmp/>`_.
        :param write_logs:      If True, write OGS log output to a file.
        :param log_level:       OGS log verbosity: ``none|error|warn|info|debug|all``.
                             ``None`` omits the ``-l`` flag (OGS default is ``info``).
        :param id:              Optional unique identifier for this execution config.

        """
        super().__init__("Execution", file_ext="yaml", id=id)

        self.interactive = interactive
        self.args = args
        self.mpi_wrapper = mpi_wrapper
        self.wrapper = wrapper
        self.ogs = ogs if ogs is not None else self._detect_ogs(mpi_ranks)
        self.mpi_ranks = mpi_ranks
        self.omp_num_threads = omp_num_threads
        self.ogs_asm_threads = ogs_asm_threads
        self.write_logs = write_logs
        self.log_level = log_level

    @classmethod
    def _detect_ogs(cls, mpi_ranks: int | None) -> str | None:
        """Detect which OGS binary or container to use based on environment and ``mpi_ranks``.

        Priority for parallel runs (``mpi_ranks`` is set):

        1. ``CONTAINER_PARALLEL`` (container with PETSc support).

        Priority for serial runs:

        1. ``OGS_BIN_PATH`` environment variable — directory containing the ``ogs`` binary.
        2. OGS Python wheel (``pip install ogs``) or ``ogs`` on PATH — returns ``None``
           so that ``ogs`` is resolved from PATH at runtime.
        3. ``CONTAINER_SERIAL`` as fallback.
        """
        from ogstools._find_ogs import has_ogs_wheel, read_ogs_path

        if mpi_ranks is not None:
            return cls.CONTAINER_PARALLEL

        ogs_path = read_ogs_path()
        if ogs_path is not None:
            return str(ogs_path)

        if has_ogs_wheel():
            return None  # ogs is on PATH via the wheel

        return cls.CONTAINER_SERIAL

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
    def from_id(cls, execution_id: str) -> Self:
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
        self._validate_thread_count()
        if dry_run:
            return [target]

        target.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "interactive": self.interactive,
            "args": self.args,
            "wrapper": self.wrapper,
            "mpi_wrapper": self.mpi_wrapper,
            "ogs": self.ogs,
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

    @property
    def env(self) -> dict[str, str]:
        """Process environment for OGS subprocess, with OMP/ASM thread vars injected."""
        e = os.environ.copy()
        if self.omp_num_threads is not None:
            e["OMP_NUM_THREADS"] = str(self.omp_num_threads)
        else:
            e.pop("OMP_NUM_THREADS", None)
        if self.ogs_asm_threads is not None:
            e["OGS_ASM_THREADS"] = str(self.ogs_asm_threads)
        else:
            e.pop("OGS_ASM_THREADS", None)
        return e

    def _value_attrs(self) -> dict[str, object]:
        """Return attributes that define value (excludes StorageBase state and env)."""
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

    @staticmethod
    def _is_container(ogs: str) -> bool:
        """Return True if ``ogs`` refers to a container image rather than a binary."""
        return ogs.startswith(("http://", "https://")) or Path(
            ogs
        ).suffix.lower() in (".sif", ".squashfs")

    @property
    def container_path(self) -> str | None:
        """Container reference for the active run, or ``None`` for native execution."""
        return (
            self.ogs
            if self.ogs is not None and self._is_container(self.ogs)
            else None
        )

    @property
    def container_prefix(self) -> str | None:
        """Container launch prefix, e.g. 'apptainer exec /path/to.sif', or None."""
        if self.container_path is None:
            return None
        is_url = self.container_path.startswith(("http://", "https://"))
        ref = (
            self.container_path
            if is_url
            else str(Path(self.container_path).expanduser())
        )
        return f"{self._CONTAINER_EXEC} {ref}"

    def _validate_thread_count(self) -> None:
        """Warn if total thread count exceeds system CPU count."""
        mpi = self.mpi_ranks or 1
        cpu_count = os.cpu_count() or 1
        omp = self.omp_num_threads or cpu_count
        if mpi * omp > cpu_count:
            warnings.warn(
                f"omp_num_threads={omp} and mpi_ranks={mpi}: "
                f"total threads ({mpi * omp}) exceeds system CPU count ({cpu_count}), "
                "which may cause resource over-subscription.",
                UserWarning,
                stacklevel=2,
            )

    def _validate_container(self) -> None:
        """Validate container path and launcher availability."""
        if self.container_path is None:
            return
        if Path(self.container_path).suffix.lower() not in (
            ".sif",
            ".squashfs",
        ):
            msg = "Container must be a *.sif or *.squashfs file."
            raise RuntimeError(msg)
        if sys.platform == "win32":
            msg = "Running OGS in a container is only possible on Linux."
            raise RuntimeError(msg)
        launcher = self._CONTAINER_EXEC.split()[0]
        if shutil.which(launcher) is None:
            msg = f"Container launcher '{launcher}' was not found."
            raise RuntimeError(msg)
        if not self.container_path.startswith(("http://", "https://")):
            container = Path(self.container_path).expanduser()
            if not container.is_file():
                msg = "Container path is not a file."
                raise RuntimeError(msg)

    @property
    def _ogs_resolved(self) -> str:
        """Resolved OGS binary path.

        Container: ogs, Path given: <Path>/ogs,
        Then try to find ogs on $PATH, falls back to ``"ogs"`` if not found.
        """
        if self.container_path is not None:
            return "ogs"
        if self.ogs is not None:
            return str(Path(self.ogs) / "ogs")
        resolved = shutil.which("ogs")
        return resolved if resolved is not None else "ogs"

    @property
    def _ogs_call_cmd(self) -> str:
        parts = []
        if self.wrapper:
            parts.append(self.wrapper)
        if prefix := self.container_prefix:
            parts.append(prefix)
        if (
            self.mpi_ranks is not None
            and self.mpi_ranks >= 1
            and self.mpi_wrapper is not None
        ):
            parts += [self.mpi_wrapper, str(self.mpi_ranks)]
        parts.append(self._ogs_resolved)
        return " ".join(parts)

    @property
    def cmd(self) -> str:
        """OGS invocation command without project file and meshes path."""
        parts = [self._ogs_call_cmd]
        if self.log_level is not None:
            parts += ["-l", self.log_level]
        if self.args is not None:
            parts.append(str(self.args))
        return " ".join(parts)

    def _validate_ogs(self) -> None:
        """Validate the OGS executable and check PETSc configuration.

        For native execution the executable must be locatable via PATH.
        Running ``ogs --version`` must succeed.
        For parallel execution (``mpi_ranks`` is set) the version output must
        contain ``-DOGS_USE_PETSC="ON"``.
        For sequential execution ``-DOGS_USE_PETSC="ON"`` must be absent.
        """
        parallel = self.mpi_ranks is not None
        version_cmd = f"{self._ogs_call_cmd} --version".strip()

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

        from ogstools._find_ogs import has_ogs_wheel

        if self.interactive and not has_ogs_wheel():
            msg = (
                "Interactive simulation requires the OGS Python wheel. "
                "Install it with: pip install ogstools[ogs]"
            )
            raise RuntimeError(msg)

    @classmethod
    def from_default(cls, file: str | Path | None = None) -> Self:
        """Create the default :class:`Execution` instance.

        If ``OGS_EXECUTION_DEFAULTS`` is set, load settings from that file and
        construct with those values. Otherwise use the empty constructor.
        """
        if file is None:
            env_path = os.environ.get("OGS_EXECUTION_DEFAULTS")
            if not env_path:
                return cls()

            path = Path(env_path)
            if not path.exists():
                msg = f"OGS_EXECUTION_DEFAULTS={env_path!r} does not exist."
                raise FileNotFoundError(msg)
        else:
            path = Path(file)

        with path.open("r") as f:
            data = yaml.safe_load(f) or {}

        return cls(**data)
