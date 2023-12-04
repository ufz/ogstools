# OGSTools 0.2.0 Release Notes

## Library

[`studies`](../user-guide/studies.md) a new package that provides utility functions to compose studies from multiple simulation
runs. For now it contains functions to perform convergence studies on simulation results (with increasing spatial/temporal discretization) for specific timesteps or over all timesteps.

[`msh2vtu`](../user-guide/msh2vtu.md) got a cleaner python interface without the need to run argparse in between.

[`feflowlib`](../user-guide/feflowlib.md) has been updated with new functionalities.
In particular, material properties can now be taken into account when converting and creating OGS models.
In addition, `feflowlib` now uses `ogs6py` to create `prj files`.
With these changes the conversion of FEFLOW models for `steady state diffusion` and `liquid flow` processes can generate a complete `prj-file` for OGS simulations.

## Tools

[`feflow2ogs`](../user-guide/feflowlib.md)-tool now enables simulation-ready `prj-files` for `steady state diffusion` and `liquid flow` processes.

## Infrastructure & Development

Web documentation for releases is now available on [ogstools.opengeosys.org](https://ogstools.opengeosys.org).
Documentation for previous releases will be available in the future and can be selected with a version dropdown in the navigation bar.
