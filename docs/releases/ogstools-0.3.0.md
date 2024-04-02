# OGSTools 0.3.0 Release Notes

## Library

[`feflowlib`](../user-guide/feflowlib.md) got:

- OGS compatible conversion of 2D meshes.
- Conversion of hydro-thermal FEFLOW models.
- Bug fix - removed `bulk_node` and `element_ids` after assignment.
- Extended `feflowlib` to be able to convert user data from FEFLOW files. User data in the FEFLOW data can store data that are not necessary for the FEFLOW simulation.

[`logparser`](../user-guide/logparser.md) got:

- Added to OGSTools with extended documentation.

[`meshlib`](../user-guide/meshlib.md) got:

- Function to compute differences of meshes.
  - The difference function from `meshlib` will now return one-to-one, pair-wise, or matrix difference depending on what input the user provides.
- Introduction of functionality to probe points on `MeshSeries`.
- Function to aggregate all timesteps in a `MeshSeries` given an aggregation function.

[`meshplotlib`](../user-guide/meshplotlib.md) got:

- Functionality and documentation for (mechanical) stress analyses.
- Both, Custom figure and axes objects, can now be passed to plot function.
- Examples are added for:
  - Custom figure axes.
  - XY labels with shared axes (Adding or not adding labels can be handled semi-automatically based on whether axes are shared).
  - Differences of meshes.
  - Limit plots.
- Progress bars: for animation and convergence study evolution evaluation.
- Label in format "property name / property unit" can be obtained from Property.
- Small fix to how setting aspect ratio is handled.
- Enable use of external `fig` and `ax` in plot functions and plotting different variables within one figure.
- Reworked aspect ratios (with examples).
- Interactive PyVista examples.

[`msh2vtu`](../user-guide/msh2vtu.md) got:

- A cleaner Python interface without the need to run argparse in between.
- A modification for `msh2vtu` to allow to convert BHE meshes.

## Infrastructure & Development

- Use latest release of OGS [ogstools.opengeosys.org](https://ogstools.opengeosys.org): 6.5.1.
- Code quality report added.
- Various changes for building OGSTools with GNU Guix.

## breaking API-Changes

- from 0.2.0 to 0.3.0

### msh2vtu

- rename parameter
- argument defaults are now the same for both CLI and python interface

```
msh2vtu(
    rdcd --> reindex (Default changed from True to False)
    ogs --> keep_ids (Reverse meaning, new default of False is the same as ogs=True before)
```

### propertylib

- rename function

```
Property(__Call__  --> transform )
```
