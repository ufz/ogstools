# monitor

```{eval-rst}
```

## Introduction

`ogsmonitor` is a Python utility that uses BokehJS to display the STDOUT log of **OpenGeoSys (OGS)**.
Internally it uses the [`logparser`](../auto_examples/howto_logparser/plot_100_logparser_intro.rst) for analysing the output.
It can be used as a standalone command line utility (see below) or within a Jupyer notebook.

## Requirements

- The monitor requires BokehJS to be installed on the system

## Command line usage

`ogsmonitor` is a command line interface of the monitor.
For example `ogsmonitor -i HM_excavation.log -j monitor.json`. The json file is not required, however it can be used to finetune the
displayed output:

### Pipe mode (recommended when starting OGS and the monitor together)

When starting OGS and `ogsmonitor` together, communicating via a log file can cause subtle timing
issues. Use a pipe instead so that OGS output flows directly into `ogsmonitor`:

```bash
ogs -l info simulation.prj | ogsmonitor
```

The OGS output is shown in the terminal and simultaneously fed to the monitor.
Press **Ctrl+C** to stop OGS and the monitor together.

To monitor, save and view the log to a file at the same time, use `tee`, which is available on Linux, macOS, and Windows PowerShell:

```bash
ogs simulation.prj | ogsmonitor -j my.json | tee my.log
```

### Post-hoc analysis

Then inspect a completed log use:

```bash
ogsmonitor -i my.log
```

```json
{
    "liveplot": true,
    "log_data": [["step_start_time", "step_size"], ["iteration_number", "dx_x_0"]],
    "time_y_axis_type": "log",
    "data_collect_time": 0.3,
    "update_plot_time": 500,
    "time_window_length": 0,
    "iteration_window_length": 0
}
```

```{argparse}
---
module: ogstools.logparser.monitor_cli
func: parser
prog: ogsmonitor
---
```
