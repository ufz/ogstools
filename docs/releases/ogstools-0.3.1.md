# OGSTools 0.3.1 Release Notes

## Overview

0.3.1 is a maintenance release (mainly bug fixes and refactorings)

Supports Python: 3.9, 3.10, 3.11, 3.12.

This is the last release with support of Python 3.9!

## OGS

6.5.2

## Feflow converter

- Conversion of component transport models with multiple components now possible
- automatic creation of OGS-6 project file template,
- bulk and boundary meshes, and
- calculation of retardation factor from sorption coefficient.
- detailed description for the example on the website.

![CT_feflow_converter.png](https://gitlab.opengeosys.org/ogs/tools/ogstools/uploads/af72b8640eb10fbefa6e8c4ccb467b34/CT_feflow_converter.png)

## Mesh: Borehole Heat Exchanger

The tool, to generate generic BHE meshes for multiple cases with simple inputs was added after the last ogstools 0.0.3 release - see [!148](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/148). The referred [!159](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/159) was only for bug fixing and small extension to the main MR 148. Here are my idea for the release notes to this feature:

- Add a tool, to generate simple ready to use BHE meshes for the HEAT-Transport-BHE Process
- automatic calculation of mesh layering
- support multiple soil layers, BHEs and groundwater layers
- support a fully 'prism' and a semi 'structured' mesh
- automatically export typical sub meshes for boundary conditions in OGS
- detailed description of an example on the website

## Function to sample properties along a user-defined polyline

- It accepts polylines instead of simple lines and list of properties defined using Property-type.

  Changes:

- introduces sample_over_polyline function to meshlib

- adds two related plots function

- in propertylib, Scalars get default color and linestyle properties that can be used in for plotting

## \[meshlib\] timevalue of min or max over timeseries

- This feature enables the visualization of when a minimum or maximum of a property happens at any point in a mesh.
- It uses the existing MeshSeries.aggregate function which can now be called with two additional "func" arguments: "min_time" and "max_time".

## Refactorings

- Moved examples data to one dedicated folder

## Bugfixes

- plotting: The streamlines in a slice of a 3D mesh are now corrected. This was due to 1) some wrong logic and 2) some floating point error when creating a Rectilineargrid to sample the values on.
