# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#


from __future__ import annotations

import abc
import copy
import errno
import inspect
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

from typing_extensions import Self


# TODO: other backup strategies could include git repo or work with provenance features of workflow managers
def _backup_move(src: Path, dst: Path) -> None:
    try:
        src.rename(dst)  # fast if on same fs
    except OSError as e:
        if e.errno == errno.EXDEV:  # cross-device link
            shutil.move(str(src), str(dst))
        else:
            raise


def _date_temp_path(
    class_id: str, suffix: str | None = None, id: str | None = None
) -> Path:
    dir = tempfile.gettempdir()

    suffix = "." + suffix if suffix else ""
    if not id:
        id = _temp_id()

    if suffix:
        return Path(dir) / class_id / id / ("default" + suffix)
    return Path(dir) / class_id / (id + suffix)


def _temp_id() -> str:
    now = datetime.now()
    return f"{now:%Y%m%d_%H%M%S_%f}"


def _check_filename_or_filepath(
    path_string: Path | str, strict: bool = True
) -> bool:
    path_string = str(path_string)
    if not path_string:
        if strict:
            msg = "The requested filepathname is empty."
            raise ValueError(msg)
        return False

    allowed_chars = set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-/\\(), :"
    )
    p = Path(path_string)

    for part in p.parts:
        if part in ("", ".", ".."):
            continue

        for c in part:
            if c not in allowed_chars:
                if strict:
                    msg = rf"The requested filepath '{path_string}' contains invalid character '{c}'. Only a-Z, 0-9, _.-\/(), and space are allowed."
                    raise ValueError(msg)
                return False

    return True


BASE_SAVE_DOC = """
Save the object.

Parameters
----------
{param_desc}
overwrite : bool
    If True, existing target will be overwritten
dry_run : bool
    If True, simulate the save without writing files
archive : bool
    If True, materialize symlinks recursively after save
"""


class StorageBase(abc.ABC):
    """
    Abstract base class for saveable objects in OGSTools.

    Provides infrastructure for:
    - Persistent storage with automatic ID generation
    - Symlink-based efficient data management
    - Backup and overwrite protection
    - Archive creation (materializing symlinks)

    Subclasses must implement:
    - _propagate_target(): Propagate save target to child objects
    - _save_impl(): Actual save logic
    - save(): Public save method (usually delegates to base implementation)
    """

    Userpath = Path("storage")  # relative paths or None
    Backup = False
    DefaultOverwrite = False  # Default value for overwrite parameter
    _SAVE_STATE_ATTRS = (
        "_id",
        "user_specified_id",
        "is_link",
        "_active_target",
        "_next_target",
        "user_specified_target",
    )

    def __init__(
        self,
        class_id: str,
        file_ext: str = "",
        default_target: Path | str | None = None,
        id: str | None = None,
    ):
        """
        Initialize a StorageBase object.

        :param class_id:        Type identifier for this saveable object
                                (e.g., "Model", "Simulation").
        :param file_ext:        File extension if this is a file (e.g., "yaml", "prj").
                                Empty string for directories.
        :param default_target:  Default path for saving if none is specified.
        :param id:              Unique identifier. If None, generates a timestamp-based ID.
        """
        self.class_id: str = class_id
        self.user_path = self.Userpath
        self.user_specified_id = bool(id)
        self._id = id or _temp_id()
        self._ext = file_ext
        self.is_link: bool = False
        new_target, user_defined = self._target_for_save(default_target, id=id)
        self._next_target: Path = new_target
        self._active_target: Path | None = (
            None  # is_saved is computed from this
        )
        self.user_specified_target: bool = user_defined

    @abc.abstractmethod
    def _propagate_target(self) -> None:
        pass

    @property
    def is_file(self) -> bool:
        """Check if this saveable represents a file (vs. directory)."""
        return self._ext != ""

    @property
    def id(self) -> str:
        """Get the unique identifier of this object."""
        return self._id

    @id.setter
    def id(self, id: str) -> None:
        """Set a new identifier and update the target path accordingly."""
        _check_filename_or_filepath(id, strict=True)
        new_target, user_defined = self._target_for_save(id=id)
        self._next_target = new_target
        self.user_specified_target = user_defined
        self.user_specified_id = True
        self._id = id
        self._propagate_target()

    def _bind_to_path(self, path: Path | str) -> None:
        """
        Bind this object to an existing file/folder on disk.

        This marks the object as saved and sets both active and next target
        to the specified path. Used when loading from existing files or
        when children are already saved before the parent.

        :param path: Path to bind this object to (file or folder).
        """
        path = Path(path)
        self.user_specified_target = True
        self._active_target = path
        self._next_target = path

    @property
    def is_saved(self) -> bool:
        """Check if this object has been saved to disk."""
        return self.active_target is not None and self.active_target.exists()

    @property
    def active_target(self) -> Path | None:
        """Get the path where this object is currently saved, or None if not saved."""
        return self._active_target

    @property
    def next_target(self) -> Path:
        """Get the path where this object will be saved next."""
        return self._next_target

    def materialize_symlinks_recursive(self, root: Path | str) -> None:
        """
        Replace all symlinks with actual copies of their targets.

        Recursively traverses the directory tree and materializes all
        symlinks, creating a standalone copy of the data.

        :param root: Path to the root directory to process.

        :raises FileNotFoundError: If the root path does not exist.
        """
        root = Path(root)
        if not root.exists():
            msg = f"{root.resolve()} does not exist"
            raise FileNotFoundError(msg)

        def _materialize(path: Path) -> None:
            """
            Replace a single path if it is a symlink.
            If folder, recurse into its contents.
            """
            if path.is_symlink():
                target = path.resolve(strict=True)
                path.unlink()
                if target.is_dir():
                    shutil.copytree(
                        target, path, symlinks=False, copy_function=shutil.copy2
                    )
                    for child in path.iterdir():
                        _materialize(child)
                else:
                    shutil.copy2(target, path)
            elif path.is_dir():
                for child in list(path.iterdir()):
                    _materialize(child)

        _materialize(root)

    def _non_default_attributes(self) -> dict[str, object]:
        """
        Used for __repr__ and __str__
        """
        signature = inspect.signature(self.__class__.__init__)
        non_defaults = {}
        for name, param in signature.parameters.items():
            if name == "self":
                continue
            if param.default is inspect._empty:
                continue
            value = getattr(self, name)
            if value != param.default:
                non_defaults[name] = value
        return non_defaults

    @staticmethod
    def _format_path(path: Path | str | None, for_repr: bool = False) -> str:
        """
        Format a Path object for display with clickable links in VS Code.

        Uses the file:// protocol for __str__ and quoted paths for __repr__.

        :param path: Path object, string, or None to format.
        :param for_repr: If True, format for __repr__ (valid Python syntax).
                        If False, format for __str__ (clickable link).
        :returns: Formatted string representation.
        """
        if path is None:
            return repr(None)
        if for_repr:
            return f"{str(path)!r}"
        return f"file://{str(path)!s}"

    def _component_status_str(self, obj: StorageBase, name: str) -> str:
        """
        Generate a status string for a child component.

        :param obj:  Child StorageBase object.
        :param name: Display name for the component.
        :returns:    Formatted status string.
        """
        if obj.is_saved:
            return f"{name}: saved to {self._format_path(obj.active_target)}"
        return f"{name}: not saved (planned to {self._format_path(obj.next_target)})"

    def _save_or_link_child(
        self,
        child: StorageBase,
        link_target: Path,
        dry_run: bool = False,
        overwrite: bool | None = None,
    ) -> list[Path]:
        """
        Save a child component or create a symlink if already saved.

        If the child is already saved, creates a symlink to avoid duplication.
        Otherwise, saves the child to its next_target location.

        :param child:       Child StorageBase object to save or link.
        :param link_target: Path where symlink should be created if linking.
        :param dry_run:     If True, simulate without writing.
        :param overwrite:   If True, allow overwriting existing files.
        :returns:           List of paths created/modified.
        """
        if child.is_saved:
            child.link(link_target, dry_run)
            return []  # because no files are created

        files = child.save(dry_run=dry_run, overwrite=overwrite)  # type: ignore[attr-defined]
        child.link(link_target, dry_run)
        return files

    def __str__(self) -> str:
        id_str = f"id: {self._id}\n"
        save_status = (
            f"  {self.__class__.__name__}(saved to: {self._format_path(self._active_target)})"
            if self.is_saved
            else f"  (not saved (planned: {self._format_path(self._next_target)}))"
        )

        return f"{self.class_id} {id_str} {save_status}"

    def __repr__(self) -> str:
        return (
            f"StorageBase("
            f"id={self._id!r}, "
            f"is_saved={self.is_saved!r}, "
            f"_active_target={self._active_target!r}, "
            f"next_target={self.next_target!r})"
        )

    def _pre_save(
        self,
        target: Path | str | None = None,
        overwrite: bool | None = None,
        dry_run: bool = False,
        id: str | None = None,
    ) -> bool:
        if target is not None and id is not None:
            msg = "Cannot specify both target and id"
            raise ValueError(msg)

        # Use class default if overwrite is None
        if overwrite is None:
            overwrite = self.DefaultOverwrite

        if id is not None:
            self.id = id
            user_defined = True
        elif target:
            _check_filename_or_filepath(target, strict=True)
            user_defined = True
            self._next_target = Path(target)
        else:
            user_defined = self.user_specified_target

        if self.is_file:
            target_overwritten = (
                self.next_target.exists() and self.next_target.is_file()
            )
        else:
            target_overwritten = (
                self.next_target.exists()
                and self.next_target.is_dir()
                and any(self.next_target.iterdir())
            )

        if user_defined and target_overwritten:
            if overwrite:
                if self._should_backup() and not dry_run:
                    backup_target = self.next_target.with_name(
                        self.next_target.name + "_backup_" + _temp_id()
                    )
                    if self.is_file:
                        backup_target.parent.mkdir(exist_ok=True, parents=True)
                    else:
                        backup_target.mkdir(exist_ok=True, parents=True)
                    _backup_move(src=self.next_target, dst=backup_target)
            else:
                type = "file" if self.is_file else "folder"
                msg = (
                    f"You are trying to overwrite the {type}: {self.next_target} "
                    f"with a new version of {self.class_id}. "
                    "If you really want this set overwrite_allowed to True."
                )
                raise ValueError(msg)

        self._propagate_target()
        return user_defined

    def _post_save(
        self, user_defined: bool, archive: bool = False, dry_run: bool = False
    ) -> None:
        if dry_run:
            return

        self.user_specified_target = user_defined
        if not user_defined and archive:
            msg = (
                f"Probably not what you want. You are trying to archive "
                f"{self._next_target}. Use save() and specify a target."
            )
            raise ValueError(msg)
        self._active_target = self._next_target
        self.is_link = False

        if archive:
            self.materialize_symlinks_recursive(self.next_target)

    def link(self, new_target: Path, dry_run: bool) -> None:
        """
        Create a symlink to the saved data at a new location.

        Efficiently creates a reference to existing saved data without
        duplicating files.

        :param new_target:  Path where the symlink should be created.
        :param dry_run:     If True, simulate without creating the link.
        """
        if dry_run:
            return
        if (
            self.active_target
            and self.active_target.exists()
            and self.active_target == new_target
        ):
            return

        if new_target.exists() or new_target.is_symlink():
            if new_target.is_file() or new_target.is_symlink():
                new_target.unlink()
            elif new_target.is_dir():
                shutil.rmtree(new_target)
            else:
                raise NotImplementedError

        assert self.active_target, "active_target should be present if saved"
        assert self.active_target.exists()
        if new_target.is_symlink():
            new_target.unlink()
        else:
            pass

        new_target.parent.mkdir(parents=True, exist_ok=True)
        new_target.symlink_to(
            Path(self.active_target.absolute()),
            target_is_directory=not self.is_file,
        )
        self.is_link = True
        return

    def _should_backup(self) -> bool:
        return StorageBase.Backup

    @staticmethod
    def saving_path() -> Path:
        """
        Get the base path for saving objects.

        :returns: Resolved user path or system temp directory.
        """
        path = StorageBase.Userpath.resolve() or tempfile.gettempdir()
        return Path(path)

    def _target_for_save(
        self,
        target: Path | str | None = None,
        next_target: Path | str | None = None,
        id: str | None = None,
    ) -> tuple[Path, bool]:

        if target:
            return Path(target), True

        if not target and next_target and self.user_specified_target:
            return Path(next_target), self.user_specified_target

        if not target and next_target:
            return Path(next_target), False

        if id:
            suffix = self._ext
            if suffix == "xdmf":
                file = "ms.pvd"  # only pvd save supported
            elif suffix == "pvd":
                file = "ms.pvd"
            elif suffix == "yaml":
                file = "meta.yaml"
            else:
                file = None

            if file:
                new_target = Path(self.user_path) / self.class_id / id / file
            else:
                new_target = Path(self.user_path) / self.class_id / id
            userspecified = True
            return Path(new_target), userspecified

        new_target = self._date_temp_path()
        userspecified = False
        return new_target, userspecified

    def _date_temp_path(self) -> Path:
        suffix = self._ext
        return _date_temp_path(self.class_id, suffix, self._id)

    def _reset_save_state(self) -> None:
        """
        Reset save-related state to initial values.

        Used after deepcopy to ensure the copy has fresh state
        (as if newly created without id or target).
        """
        self._id = _temp_id()
        self.user_specified_id = False
        self._active_target = None
        self.is_link = False
        new_target, _ = self._target_for_save(None)
        self._next_target = new_target
        self.user_specified_target = False

    def copy(
        self,
        target: Path | str | None = None,
        id: str | None = None,
        deep: bool = True,
    ) -> Self:
        """
        Performs a deepcopy and reassign the associated files/ids. Use this function together with the constructor to tell, that it is your intent to protect the file, you used for creating the object.
        e.g. prj1 = Project(file1).copy(id="prj1"), here you want to protect the original file1.

        :param target:  If provided, the target is the location for the next call of save(). Mutually exclusive with id.
        :param id:      If provided, set the id for the next call of save().
        :param deep:    switch to choose between deep (default) and shallow
                        (self.copy(deep=False)) copy.
        :returns:      A deep copy of this object associated to a new file/folder or id
        """
        if target is not None and id is not None:
            msg = "Cannot specify both target and id"
            raise ValueError(msg)

        new_instance = copy.deepcopy(self) if deep else copy.copy(self)

        if id is not None:
            new_instance.id = id
            # new_instance.user_specified_id = True
            # new_target, user_defined = new_instance._target_for_save(id=id)
            # new_instance._next_target = new_target
            # new_instance.user_specified_target = user_defined

        elif target is not None:
            new_instance._next_target = Path(target)
            new_instance.user_specified_target = True
            new_instance._propagate_target()
        return new_instance
