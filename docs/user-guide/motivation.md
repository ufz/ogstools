# Motivation

## Problem statement

Python is utilized in numerous [OpenGeoSys(OGS)](https://www.opengeosys.org)-related projects, spanning both the preparation and assessment of simulation results. Our positive experience with [ogs6py](https://github.com/ufz/ogs6py) and [VTUInterface](https://github.com/ufz/vtuinterface) highlights the community's strong reliance and extensive usage of these tools. However, the identified need for a wider spectrum of functionalities remains, essential for catering to the diverse requirements across various projects. OGSTools aims to address this need by expanding functionalities to accommodate past requirements and current project-specific needs.

## Target audience

OGSTools is for OpenGeoSys users and developers aiming to effectively automate their pre and post-processing tasks (workflows). Proficiency in basic Python, accompanied by knowledge of essential libraries like [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [Matplotlib](https://matplotlib.org/) in addition to familiarity with [OpenGeoSys - Benchmarks](https://www.opengeosys.org/docs/benchmarks/), constitutes the required skill level.

## Functionality and unique features

OGSTools consists of features designed specifically for [OpenGeoSys](https://www.opengeosys.org) but can be applied broadly across multiple [OpenGeoSys](https://www.opengeosys.org)-specific projects. The functionality is grouped  thematically into sub-libraries that are developed to  collaborate with each other. 

## Versatility and Flexibility
All Sub-Libraries either 
- transform from [OpenGeoSys](https://www.opengeosys.org) specific data into data structure of common python libraries
- transform from data structures of common libraries to [OpenGeoSys](https://www.opengeosys.org) specific data
This compatibility enables [OpenGeoSys](https://www.opengeosys.org) users to harness the full potential of Python's extensive ecosystem.

## Community Support and Maintenance

OGSTools is part of the core development (with a dedicated team of developers). Progress is  mainly made by project-specific requests. It is ready for broader adoption with the OGS Community.

## Use Cases or Case Studies
- [AREHS](https://www.ufz.de/index.php?en=47155) - within a [snakemake](https://snakemake.readthedocs.io) based complex workflow
- [OpenWorkflow](https://www.ufz.de/index.php?en=48378) - within a workflow for thermal dimensioning of a deep geological repository
- [OpenGeoSys - Benchmarks](https://www.opengeosys.org/docs/benchmarks/)

## Future Development and Roadmap
- 2024:
	- integration of existing [OGS log file parser](https://github.com/joergbuchwald/ogs6py/tree/master/ogs6py/log_parser)
	- integration of [ogs6py](https://github.com/ufz/ogs6py)
	- integration of [VTUInterface](https://github.com/ufz/vtuinterface)
	- scalability study