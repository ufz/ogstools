# SPDX-FileCopyrightText: Copyright (c) OpenGeoSys Community (opengeosys.org)
# SPDX-License-Identifier: BSD-3-Clause

import subprocess
from collections.abc import Callable, Collection
from pathlib import Path
from typing import Any, ClassVar

from .provide_ogs_cli_tools_via_wheel import binaries_list, ogs_with_args


class CLI_ON_PATH:

    def __init__(self, ogs_bin_dir: Path):
        self.ogs_bin_dir = ogs_bin_dir
        for b in binaries_list:
            if b == "ogs":
                continue  # command provided separately
            setattr(self, b, CLI_ON_PATH._get_run_cmd(ogs_bin_dir, b))

    def ogs(self, *args: Any, **kwargs: Any) -> int:
        """
        This function wraps the commandline tool ogs for easy use from Python.

        It returns the return code of the commandline tool.

        The entries of args are passed as is to the commandline tool.
        The entries of kwargs are transformed: one-letter keys get a single
        dash as a prefix, multi-letter keys are prefixed with two dashes,
        underscores are replaced with dashes.

        Thereby, commandline tools can be used in a "natural" way from Python, e.g.:

        >>> cli = CLI()
        >>> cli.ogs("--help") # prints a help text
        ...
        >>> cli.ogs(help=True) # flags without any additional value can be set to True
        ...

        A more useful example. The following will create a line mesh:

        >>> outfile = "line.vtu"
        >>> cli.generateStructuredMesh(e="line", lx=1, nx=10, o=outfile)
        """

        cmdline = CLI_ON_PATH._get_cmdline("ogs", *args, **kwargs)

        if self.ogs_bin_dir:
            print("OGS_USE_PATH is true: ogs from $PATH is used!")
            return subprocess.call(cmdline)

        return ogs_with_args(cmdline)

    #  Tools that already use underscores in the CLI
    # (e.g. --geometry_name is really the argument).
    _UNDERSCORE_ARGS: ClassVar[dict[str, set[str]]] = {
        "addDataToRaster": {
            "ll_x",
            "ll_y",
            "ur_x",
            "ur_y",
            "offset_value",
            "scaling_value",
            "output_raster",
        },
        "AddElementQuality": {
            "input_mesh_file",
            "output_mesh_file",
            "quality_criterion",
        },
        "checkMesh": {"print_properties"},
        "createLayeredMeshFromRasters": {"ascii_output"},
        "createRaster": {"cell_size", "ll_x", "ll_y", "n_cols", "n_rows"},
        "createTetgenSmeshFromRasters": {"ascii_output"},
        "generateGeometry": {"geometry_name", "polyline_name"},
        "geometryToGmshGeo": {
            "average_point_density",
            "max_points_in_quadtree_leaf",
            "mesh_density_scaling_at_points",
            "mesh_density_scaling_at_stations",
            "write_merged_geometries",
        },
        "identifySubdomains": {"output_prefix"},
        "mergeMeshToBulkMesh": {
            "capillary_pressure",
            "gas_pressure",
            "material_id",
            "sigma_xx",
            "sigma_yy",
            "sigma_zz",
        },
        "NodeReordering": {"input_mesh", "output_mesh", "no_volume_check"},
        "OGS2VTK": {"ascii_output"},
        "partmesh": {"exe_metis"},
        "pvtu2vtu": {"original_mesh"},
        "ResetPropertiesInPolygonalRegion": {"any_of"},
        "vtkdiff": {"first_data_array", "second_data_array", "mesh_check"},
        "xdmfdiff": {"first_data_array", "second_data_array", "mesh_check"},
    }

    @staticmethod
    def _format_kv(kwargs: Any, underscore_args: Collection[str] = ()) -> Any:
        for key, v in kwargs.items():
            # Convert None to True
            if v is None:
                # TODO: Remove after 2025/08
                print(
                    f"Deprecation warning: Setting {v} to `None` is deprecated, set to `True` instead!"
                )
                v = True  # noqa: PLW2901

            # If value is False then ignore
            if isinstance(v, bool) and not v:
                continue

            if len(key) == 1:
                yield f"-{key}"
            else:
                flag = key if key in underscore_args else key.replace("_", "-")
                yield f"--{flag}"

            # Pass value if not bool
            if not isinstance(v, bool):
                yield f"{v}"

    @staticmethod
    def _get_cmdline(
        cmd: str,
        *args: Any,
        underscore_args: Collection[str] = (),
        **kwargs: Any,
    ) -> list[str]:
        str_kwargs = list(CLI_ON_PATH._format_kv(kwargs, underscore_args))
        return [cmd] + str_kwargs + list(args)

    @staticmethod
    def _get_run_cmd(ogs_bin_dir: Path, attr: str) -> Callable:
        underscore_args = CLI_ON_PATH._UNDERSCORE_ARGS.get(attr, set())

        def run_cmd(*args: Any, **kwargs: Any) -> int:
            cmd = str(ogs_bin_dir / attr)
            stdout = kwargs.pop("stdout", None)
            stderr = kwargs.pop("stderr", None)
            if stdout is False:
                stdout = subprocess.DEVNULL
            else:
                print("CMD:", cmd)
            if stderr is False:
                stderr = subprocess.DEVNULL
            cmdline = CLI_ON_PATH._get_cmdline(
                cmd, *args, underscore_args=underscore_args, **kwargs
            )
            return subprocess.call(cmdline, stdout=stdout, stderr=stderr)

        # TODO: Only arguments with underscores work. Arguments with dashes not.
        run_cmd.__doc__ = f"""
            This function wraps the commandline tool {attr} for easy use from Python.

            It returns the return code of the commandline tool.

            The entries of args are passed as is to the commandline tool.
            The entries of kwargs are transformed: one-letter keys get a single
            dash as a prefix, multi-letter keys are prefixed with two dashes,
            underscores are replaced with dashes.

            Thereby, commandline tools can be used in a "natural" way from Python, e.g.:

            >>> cli = CLI()
            >>> cli.{attr}("--help") # prints a help text
            ...
            >>> cli.ogs(help=True) # flags without any additional value can be set to True
            ...

            A more useful example. The following will create a line mesh:

            >>> outfile = "line.vtu"
            >>> cli.generateStructuredMesh(e="line", lx=1, nx=10, o=outfile)
            """

        return run_cmd
