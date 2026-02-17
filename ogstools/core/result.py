# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

import filecmp
from pathlib import Path

from .storage import StorageBase


class Result(StorageBase):
    """
    Container for OGS simulation results and output files.

    Manages the simulation output directory containing result files
    (meshes, logs, etc.) and provides access to simulation outputs.
    """

    def __init__(
        self,
        sim_output: Path | str | None = None,
    ) -> None:
        """
        Initialize a Result object.

        :param sim_output:  Path to the simulation output directory.
                            If None, uses a default location.
        """
        super().__init__("Result")

        if sim_output:
            self.sim_output = Path(sim_output)
            self._bind_to_path(sim_output)
        else:
            self.sim_output = self.next_target

        self._log_filename = "log.txt"

        self.next_target.mkdir(parents=True, exist_ok=True)

    @property
    def log_file(self) -> Path:
        """Get the path to the log file, following the current target location."""
        base = self.active_target or self.next_target
        return base / self._log_filename

    def _save_impl(self, dry_run: bool = False) -> list[Path]:

        target = self.next_target

        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            if self.active_target and target != self.active_target:
                target.symlink_to(self.active_target, target_is_directory=True)

        # Return the target directory that was created/linked
        return [self.next_target]

    def _propagate_target(self) -> None:
        self.sim_output = self.next_target

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Result):
            return False
        return (
            filecmp.dircmp(self.sim_output, other.sim_output).diff_files == []
        )

    def save(
        self,
        target: Path | str | None = None,
        overwrite: bool | None = None,
        dry_run: bool = False,
        archive: bool = False,
        id: str | None = None,
    ) -> list[Path]:
        """
        Save or link the result output directory.

        Creates a symlink to the actual simulation output directory at the
        target location. Use archive=True to copy all data instead.

        :param target:      Path where the result should be saved/linked.
                            If None, uses a default location.
        :param overwrite:   If True, overwrite existing target. Defaults to False.
        :param dry_run:     If True, simulate save without creating files.
        :param archive:     If True, copy all data instead of creating symlinks.
        :param id:          Optional identifier. Mutually exclusive with target.
        :returns:           List of Paths to created files/directories.
        """
        user_defined = self._pre_save(target, overwrite, dry_run, id=id)
        files = self._save_impl(dry_run=dry_run)
        self._post_save(user_defined, archive, dry_run)
        return files

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        base_repr = super().__repr__()

        output_str = self.active_target if self.is_saved else self.sim_output
        construct = f"{cls_name}(sim_output={str(output_str)!r})"

        return f"{construct}\n{base_repr}"

    def __str__(self) -> str:
        base_str = super().__str__()
        lines = [base_str]
        lines.append(f"  - sim_output: {self._format_path(self.sim_output)}")
        lines.append(f"  - log_file: {self._format_path(self.log_file)}")
        return "\n".join(lines)
