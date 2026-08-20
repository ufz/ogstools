# Third-party applications

```{eval-rst}
```

`ogstools` makes use of several third party python packages:

- [tqdm](https://github.com/tqdm/tqdm): progress bars in computionally
  expensive loops
  - if you want to globally deactivate them please set the following environt
    variable: `os.environ["TQDM_DISABLE"] = "1"`

It also relies on the following external (non-pip) applications:

- [Tetgen](https://wias-berlin.de/software/tetgen/): optional, used by
  {py:meth}`ogstools.mesh.create.LayerSet.to_region_tetrahedron` to build
  tetrahedral meshes. Tetgen is not bundled with `ogstools` and is not
  installed via `pip` — it must be available on the system `PATH`.

  | Version | Linux | Windows | macOS |
  | ------- | ------------ | -------- | -------- |
  | 1.6.0 | tested in CI | untested | tested once |
  | 1.5.1 | tested once | untested | untested |
  | 1.5.0 | tested once | untested | untested |

  Only Linux + Tetgen 1.6.0 is checked continuously (installed by CI from
  conda-forge, unpinned).

  If you need a version/platform combination other than Linux + 1.6.0
  confirmed or maintained, please ask on the
  [OpenGeoSys Discourse forum](https://discourse.opengeosys.org/c/usability/8).
