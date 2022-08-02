# Documentation of ogsdatacompare

Overview of functions in this package can be found in this document.

## What has to be done manually?
Things like axis captions, need to be modified in the code directly.

## General setup

In the main file there has to be defined a path to folder containing results of experiments, that are supposed to be compared and plotted. Each experiment has to be contained inside of a separate folder. Inside the main result folder, a subfolder called `Results` has t

### Naming convention
There is a function extract_params, that can be used to automatically extract parameter names and values from folder names. In order to use this function, each experiment has to be named according to following convention:
```
ParameterName1_parametervalue1_ParameterName2_parametervalue2
```
Currently only ogd_compare_3d_view requires it, but it can be added to other functions as well.
The experiment folder has to contain one pvd file. Other files will be ignored. Folders without pvd file will be skipped.

### Configuration dictionary
Most of the functions require a configuration dictionary to run. What exactly has to be in it, varies between functions. See an example below:

```
settings = {'color_ref': 'red',
            'colors': ['blue', 'orange', 'teal', 'violet', 'purple', 'bisque',
                       'gold', 'yellowgreen', 'aqua',  'indigo', 'grey', 'olive',
                       'sienna', 'green', 'salmon', 'fuchsia', 'black', 'orchid',
                       'lime', 'azure', 'orangered', 'coral', 'turquoise', 'wheat',
                       'ivory', 'brown', 'crimson', 'magenta', 'maroon', 'silver',
                       'tan'],
            #'colors': cm.rainbow(np.linspace(0, 1, 30)),
            'compare': ['displacement'],
            'skip': ['temperature', 'pressure', 'p0'],
            'plot_ext': ['png'],
            'points_file': os.path.join(os.getcwd(), 'ogs_compare/Tools', 'points_isotropic.csv'),
            'include_legend': True
            }
```
At the end of the ogs_compare/ogs_compary.py file, a function can be found that reads this dictionary. It gives an overview of what parameters are available and which of them are mandatory.

### Warning
Methods read_data and read_data_full assume that the input data is 2D. For 3D and 1D files, "dim" parameter needs to be adjusted in the code. In the future it may be done automatically.

## Main plotting functions

### ogs_compare_time_point
#### Syntax
```python
OGSDataCompare.ogs_compare_time_point(time_step)
```
Configuration dictionary is explained in "General setup" section, result_folder is an access path (both absolute and relative should work) to folder containing all experiments, time_step is a list of time steps at which the results will be compared given as an array. Each value in the time_step array will generate a separate figure.

#### Goal
Creates plot with value of an output parameter (temperature, pressure, etc) on Y-axis and separate experiments on X-axis.

### ogs_compare
#### Syntax
```python
OGSDataCompare.ogs_compare()
```
This function doesn't require any input.
Optional parameters:
- analytical_ref
#### Goal
Creates a figure with separate subfigures for all observation points. Each of this figures plots parameter vs time functions for all experiments in one subplot. This function is not fully automatic. The number of subplots, position of legend etc are setup for data I was working with. For other datasets, it may require some tweaking in the code.

### ogs_compare_separate
#### Syntax
```python
OGSDataCompare.ogs_compare_separate()
```
#### Goal
The same as ogs_compare but all subfigures are plotted and saved as separate figures and files.

### ogs_compare_3D_view
#### Syntax
```python
OGSDataCompare.ogs_compare_3d_view()
```
#### Goal
Creates a 3D comparing the output parameter for all combinations of two tested parameters in the provided experiments. Each plot is for one time step and observation point combination. This function requires experiment folders being named in accordance with the convention discussed in "Naming convention" section.

### plot_min_max
##### Syntax
Defeault read_data() method only read values at specified points. read_data_full() needs to be called before plot_min_max().
```python
OGSDataCompare.read_data_full()
OGSDataCompare.plot_min_max()
```
#### Goal
Plot minimal and maximal values of a specified parameter in each time step. Parameter values are displayed on primary and secondary y-axis and time is on the x-axis.
