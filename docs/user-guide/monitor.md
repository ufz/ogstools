# monitor

```{eval-rst}
```

## Introduction

`ogsmonitor` is a Python utility that uses BokehJS to display the STDOUT log of **OpenGeoSys (OGS)**.
Internally it uses the [`logparser`](../auto_examples/howto_logparser/plot_100_logparser_intro.rst) for analysing the output.
It can be used as a standalone command line utility (see below) or within a [`Jupyer notebook`](../auto_examples/howto_simulation/plot_010_simulate.rst) class it is possible to create a proposal of a `prj-file` from the converted model to enable simulations with OGS.
At the moment `steady state diffusion`, `liquid flow`, `hydro thermal` and `component/mass transport` processes are supported to set up complete `prj-files`.
For other processes, a generic `prj-file` is created that needs manual configurations to be ready for OGS simulation.

## Requirements

- The monitor requires BokehJS to be installed on the system

## Command line usage

`ogsmonitor` is a command line interface of the converter that summarizes the main functions to provide the user with an accessible application.

```{argparse}
---
module: ogstools.logparser.monitor_cli
func: parser
prog: ogsmonitor
---
```
