# Patterns

Each section describes one family of components: its members, the skeleton that every member has, and the allowed variations.
A new member follows the skeleton.
A member that deviates without an allowed reason is a defect.

## File-backed objects

Members: `Project`, `Model`, `Simulation`, `Result`, `Execution`, `Meshes`, `MeshSeries`.
All derive from `ogstools.core.storage.StorageBase`.

Skeleton:

- `__init__(..., id: str | None = None)` calls `super().__init__("<ClassName>", file_ext, id=id)`.
  The class name is the storage sub-folder.
- Classmethod constructors return `Self`:
  - `from_id(id)` loads from the user's storage folder.
  - `from_file(path)` loads an object stored as one file.
  - `from_folder(path)` loads an object stored as a directory.
  - Further `from_<source>` constructors exist where users have other sources (`Meshes.from_gmsh`, `MeshSeries.from_data`).
- `copy(target=None, id=None, deep=True)` is inherited.
  A subclass customises copying through `__deepcopy__`.
- `save(target: Path | str | None = None, overwrite: bool | None = None, dry_run: bool = False, archive: bool = False, id: str | None = None) -> list[Path]`.
  Extra parameters come after these, as keywords.
  The body is `_pre_save` → `_save_impl(dry_run, ...)` → `_post_save`.
- `__repr__` returns a constructor call that can be evaluated (`Cls.from_id(...)`, `Cls.from_folder(...)` or `Cls(...)`), followed by `super().__repr__()`.
- `__str__` starts with `super().__str__()`.
- `__eq__` compares content and sets `__hash__ = None`.
- Constructing an object changes neither the file system nor global state.
  Writing happens in `save`.
- A file-backed object has one write path, `save`.

Allowed variations:

- `Result` is always bound to an existing output directory and has no `id`.
- `Simulation` takes no `id` in the constructor. It gets one from `save(id=...)` or `from_id`.
- `ReferencedFile` and its subclasses are created only by `Project`. They have no `from_*`, and `__repr__` and `__str__` show only the file name.
- `Project(input_file=...)`, `MeshSeries(filepath)` and `Model(project=...)` also take the path in the constructor. They keep it for compatibility.
  A new member constructs from a path only through the classmethods.

## Command-line tools

Members: `msh2vtu`, `feflow2ogs`, `yaml2vtu`, `ogsmonitor` (the `[project.scripts]` entries).

Skeleton:

- `argparser() -> argparse.ArgumentParser` builds the parser.
- `cli() -> int` parses the arguments, calls one library function and returns the exit code.
- The tool lives in `<subpackage>/_cli.py`, one tool per subpackage.
  The `[project.scripts]` entry points to `ogstools.<subpackage>._cli:cli`.
- The tool is a thin wrapper. All logic lives in the library, and the tool only maps arguments.
- `-l/--log` turns on verbose logging.
  The tool does not configure logging on import.
- The tool prints its results and errors.
  On failure it returns a non-zero exit code instead of raising.
- A test calls every entry point at least with `--help`.

Allowed variations:

- `ogsmonitor` and `yaml2vtu` live in `logparser/monitor_cli.py` and `meshes/meshes_from_yaml_cli.py`.

## Plot functions

Members:

- the functions in `ot.plot` (`contourf`, `subplot`, `heatmap`, `line`, `quiver`, `streamlines`, `shape_on_top`, `animate`),
- the plot methods (`MeshSeries.plot_line`, `MeshSeries.plot_time_slice`, `Meshes.plot`, `Model.plot_constraints`, `Log.plot_convergence`),
- the convergence-study plots.

Skeleton:

- Data first, then the variable: `f(mesh, variable, ...)`.
- Styling options are documented `**kwargs`.
  Their defaults are read from `ot.plot.setup` at call time.
- A function that cannot draw the given input raises `ValueError` with the remedy.

Allowed variations:

- `contourf_pv`, and `contourf` with `interactive=True`, return a `pv.Plotter`, because they plot with the pyvista backend.
- `animate(fig, plot_func)` takes a figure and a plot function instead of data, and returns a `FuncAnimation`.
- `MeshSeries.plot_time_slice(x, y, variable)` takes the axes quantities before the variable.

How a function receives and returns the figure:

- A function that draws one plot is figure-level (`contourf`, `heatmap`, `line`, `MeshSeries.plot_time_slice`).
  `fig=None, ax=None` are optional (`line` takes only `ax`). Without them, the function creates and returns a `Figure`.
  With them, it draws into them and returns `None`.
- A function that draws a report with several axes is report-level (log-parser and convergence-study plots, `Meshes.plot`, `Model.plot_constraints`).
  It always creates and returns a new `Figure`.
- A new axes-level function (a required `ax` after the data, returns `None`) is an internal helper.
  The existing public axes-level functions `subplot`, `quiver`, `streamlines` and `shape_on_top` stay public.

## Variables and units

Members: `Variable` and its subclasses `Scalar`, `Vector`, `Matrix`, `Components_BHE`, `VectorList`, their predefined instances, and the unit registry `u_reg`.

Skeleton:

- A public function that takes a variable accepts `Variable | str` and resolves it once with `Variable.find`.
- A preset is built with keyword arguments: `Scalar(data_name=..., data_unit=..., output_unit=..., output_name=..., symbol=..., mask=..., cmap=...)`.
- A preset of a physical quantity has `data_unit` and `output_unit`.
  A categorical preset (`material_id`) has none.
- Similar presets can be made with `<preset>.replace(...)`.
- Derived variables (`.magnitude`, `.difference`, `.abs_error`) are read-only properties that return a new `Variable`.
  They are made with `Variable.from_variable(<preset>, ...)`.
- Units are pint strings (`"K"`, `"°C"`, `"a"`).
  A conversion goes through the `output_unit` of a `Variable` or through `ogstools.variables.u_reg`.
  It never uses a hard-coded factor (`/ 86400`, `* 3.1536e7`, `* 1e-6`).

Allowed variations:

- A self-contained physics model (`ogstools.physics.nuclearwasteheat`) may keep its own registry.
  Its quantities must never meet `u_reg` quantities.

## Mesh generators and converters

Members:

- `ot.gmsh_tools.rect`, `cuboid`, `remesh_with_triangles`, and `ogstools.meshes.gmsh_BHE.gen_bhe_mesh`,
- the gmsh and YAML converters behind `Meshes.from_gmsh` and `Meshes.from_yaml`,
- the FEFLOW converters (`read_feflow`, `FeflowModel`).

Skeleton:

- Users reach a converter through a classmethod of the class it produces (`Meshes.from_gmsh(ot.gmsh_tools.rect(...))`).
  The converter function itself is internal.
- The input path is `Path | str`.
- A generator whose tool writes a file (gmsh) returns the `Path` of that file.
  Its output parameter is `output_file` for a file and `output_dir` for a directory.
  `None` means a file under `ogstools.definitions.temp_dir()`.
- A reader returns the in-memory object (`Meshes`, `pv.UnstructuredGrid`).

Allowed variations:

- `FeflowModel` stays a converter with its own `save`, not a file-backed object.
  It is not stored and reloaded, and FEFLOW is not on PyPI.

## Example data

Members: the loader functions and path constants in `ogstools.examples`.

Skeleton:

- A loader is `load_<type>_<name>() -> <Type>` (`load_meshseries_THM_2D_PVD() -> MeshSeries`).
  `<type>` is the returned class in snake case, and `mesh` for a `pv.UnstructuredGrid`.
  Its body calls the public constructor on a file below `EXAMPLES_DIR`.
- A loader returns a new object, so callers can change it.
  For a file-backed object it returns a copy.
- A path constant is `<kind>_<name>`, with the file kind as prefix (`prj_mechanics`, `log_parallel`, `msh_square`).

## Project-file builders

Members: the `BuildTree` components that `Project` exposes as attributes (`prj.time_loop`, `prj.parameters`, `prj.media`, …).

Skeleton:

- A component subclasses `BuildTree` and is constructed with the shared XML tree.
  Users reach it only through its `Project` attribute.
- Mutators are `add_<element>` (append a child) and `set_<element>` (create or replace).
  They return `None`.
- Options that users normally set are typed parameters.
  `**kwargs` is used only where the OGS XML schema is open-ended. The docstring names the accepted keys.

## Log-parser records

Members: the record classes and regex tables in `ogstools.logparser`, and the analyses wrapped by `ot.Log`.

Skeleton:

- A record is a `@dataclass` derived from its context mixin and `MPIProcess`.
  Its fields are named after the DataFrame columns, in the order of the regex groups.
- A regex table is a list of pairs of a raw-string pattern and exactly one record class.
  The pattern starts with the level prefix (`info:`, `error:`) and escapes literal characters.
- An analysis takes the parsed `pd.DataFrame`, checks its input columns, builds a pivot table and checks its output columns.
  Users reach it as a method of `ot.Log`.

Allowed variations:

- One regex table per OGS log-format version.
