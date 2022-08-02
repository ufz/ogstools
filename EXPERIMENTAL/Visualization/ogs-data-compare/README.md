# ogsdatacompare

Python scripts that create different plot comparing results of OGS simulations

## Usage
There are two ways of using ogs-data-compare.

You can provide a dictionary with settings:
```python
settings = { # Required parameters
            'parameter': 'parameter_to_observe',
            'path': 'path/to/experiment/folder', # os.path.join is recommended
            'points_file': 'path/to/observation/points/definition/file.csv',  # os.path.join is recommended

            # Optional parameters
            'plot_ext': ['png'],
            'include_legend': True
            }

data_compare = ogsdatacompare.OGSDataCompare(settings)
```
or provide mandatory arguments later, when read_data() method is called:
```python
data_compare = ogsdatacompare.OGSDataCompare()
data_compare.read_data(path='path/to/experiment/folder',  # os.path.join is recommended
                       parameter='parameter_to_observe',
                       points_path='path/to/observation/points/definition/file.csv'  # os.path.join is recommended
                      )
```

Those two methods can be combined. For example settings dictionary can contain definitions for some optional parameters and mandatory ones can be defined when read_data() method is called.

## Manual
For more detailed description, please see Documentation/ogs_compare.md or the example jupyter notebook in Documentation/Example.
