# meshlib

## Overview

```
{eval-rst}
.. sectionauthor:: Tobias Meisel (Helmholtz Centre for Environmental Research GmbH - UFZ)
```

`meshlib` is a versatile Python library designed for efficient mesh generation from surfaces. It offers a rich set of features that include supporting various meshing algorithms(e.g. prism, tetraeder, voxel in 3D), which can be customized through a range of parameters to suit specific requirements. `meshlib` enables the creation of meshes tailored for studies involving convergence or scalability, providing researchers and engineers with valuable insights into their simulations.

A unique aspect of `meshlib` is its seamless integration with both pyvista and ogs command line tools, allowing users to visualize and analyze generated meshes effortlessly. These meshes are suitable for finite element method (FEM) calculations using OpenGeoSys (OGS).

## Getting started

Following examples demonstrate the usage of the meshlib:

- [](../auto_examples/howto_preprocessing/plot_meshlib_pyvista_input.rst)
- [](../auto_examples/howto_preprocessing/plot_meshlib_vtu_input.rst)

You can access the comprehensive API documentation at: [](../reference/ogstools.meshlib).
