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
