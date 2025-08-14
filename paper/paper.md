---
title: 'OGSTools: A Python package for OpenGeoSys'
tags:

- Python
- OpenGeoSys
- Workflows
- finite element method
- geosciences
- Computer simulation
- preprocessing
- modelling
- postprocessing
- interface
- open-source
authors:
- name: Tobias Meisel
  orcid: 0009-0009-8790-8903
  equal-contrib: true
  affiliation: 1
- name: Florian Zill
  orcid: 0000-0002-5177-401X
  equal-contrib: true
  affiliation: 2, 1
- name: Julian Heinze
  orcid: 0009-0004-3449-8852
  equal-contrib: true
  affiliation: 1
- name: Lars Bilke
  orcid: 0000-0001-8986-2413
  equal-contrib: true
  affiliation: 1
- name: Max Jäschke
  orcid: 0009-0005-2196-0830
  equal-contrib: true
  affiliation: 3, 1, 4
- name: Feliks Kiszkurno
  orcid: 0000-0003-3304-4838
  equal-contrib: true
  affiliation: 2, 1
- name: Jörg Buchwald
  orcid: 0000-0001-5174-3603
  equal-contrib: true
  affiliation: 1

affiliations:

- name: Helmholtz Centre for Environmental Research, Germany
  index: 1
  ror: 000h6jb29
- name: TU Bergakademie Freiberg, Germany
  index: 2
  ror: 031vc2293
- name: Leipzig University of Applied Sciences, Germany
  index: 3
  ror: 03xgcq477
- name: Technische Universität Dresden, Germany
  index: 4
  ror: 042aqky30

date: 17 March 2025
bibliography: paper.bib

______________________________________________________________________

## Summary

`OGSTools` (`OpenGeoSys` Tools) is a Python library for streamlined usage of `OpenGeoSys 6` (OGS) - a software for simulating thermo-hydro-mechanical-chemical (THMC) processes in porous and fractured media \[@bilke_2025_14672997\] \[@kolditz2012opengeosys\].
`OGSTools` \[@ogstools2025\] provides an interface between OGS-specific data and well-established data structures of the Python ecosystem, as well as domain-specific solutions, examples, and tailored defaults for OGS users and developers. By connecting OGS to the ecosystem of Python, the entry threshold to the OGS platform is lowered for users with different levels of expertise.
The libraries' functionalities are designed to be used in complex automated workflows (including pre- and post-processing), the OGS benchmark gallery, the OGS test-suite, and in automating repetitive tasks in the model development cycle.

## Statement of need

### Development efficiency

Modellers of OGS iteratively run simulations, analyse results, and refine their models with the goal to enhance the accuracy, efficiency and reliability of the simulation results.
To improve efficiency, repetitive steps in the model development cycle should be formalized and implemented - in the case of OGSTools as a Python library.
The Python library introduced here serves as a central platform to collect and improve common functionalities needed by modellers of OGS.

### Complex workflows

A workflow is a structured sequence of steps, that processes data and executes computations to achieve a specific goal \[@diercks2022workflows\].
In our scientific research, workflows need to integrate multiple steps—such as geological data preprocessing, ensemble simulations with OGS, domain-specific analysis and visualization—into complex and fully automated and therefore reproducible sequences.
Typically, one specific workflow is implemented to answer one specific scientific question.
Workflow-based approaches have been proven to adhere to the `FAIR principles` \[@goble2020fair\], \[@Wilkinson_2025\]. The typical approach is to use existing workflow management software that covers domain-independent parts like dependency graph description, computational efficiency, data management, execution control, multi-user collaboration and data provenance \[@Bilke2025\].
When building upon the Python ecosystem, Python-based workflow managers such as `Snakemake` \[@Köster2012\] and `AiiDA` \[@Huber2020\] are a natural choice.

The usage of workflow managers shifts the actual development to the workflow components.
Common and frequently used functionality found within workflow components is made reusable and provided in this Python library.
It focuses on functionalities directly related to (1) the OGS core simulator and its specific input and output data, (2) domain-specific definitions in geo-science, (3) finite element modelling (FEM), and (4) numerical computation.
The workflow components are constructed from generic Python libraries, OGSTools, and integration code tailored to the respective workflow manager chosen.
To ensure integration of the library's functionalities with the workflow management software, a list of functional and non-functional requirements (e.g., thread safety), imposed by workflow management software, is maintained and regularly validated through continuous testing.

### Test suite

`OGSTools` provides functionality for (1) setting up test environments, (2) executing OGS under specified conditions, (3) evaluating OGS results against defined test criteria, and (4) monitoring the testing process.

### Educational Jupyter notebooks

OGS is already being used in academic courses and teaching environments. With Jupyter Notebooks, students can explore interactive learning environments where they directly modify parameters, material laws, and other influencing factors, and instantly visualize the outcomes. `OGSTools` serves as an interface between OGS and Jupyter Notebooks. It supports the creation of input data—such as easily configurable meshes or ready-to-use project files.

### Centralization of Python-related development

Previously, the code base for Python-related tasks in OGS was fragmented, with components that were often developed for specific use cases, with varying degrees of standardization and sharing.
The lack of centralization led to inefficiencies, inconsistent quality, and challenges in maintaining and extending the code.
With `OGSTools`, reusable Python code is now centralized under the professional maintenance of the core developer team of OGS.
It ensures better collaboration, standardized practices, and improved code quality.
Further, it enables the transfer of years of experience in maintaining the OGS' core \[@Bilke2019\] to the pre- and post-processing code.
For the centralized approach, preceding work from `msh2vtu` \[@msh2vtu\], `ogs6py and VTUInterface` \[@Buchwald2021\] and extracted functionalities from the projects (1) `AREHS` \[@arehs2024\], and (2) `OpenWorkFlow - Synthesis Platform` \[@openworkflow2023\] have been adapted and integrated into `OGSTools`.

To address `The Need for a Versioned Data Analysis Software Environment` \[@Blomer2014\] `OGSTools` provides additionally a pinned environment, updated at least once per release.
While reproducibility requires environments with pinned dependencies, `OGSTools` is additionally tested with the latest dependencies, to receive early warnings of breaking changes and to support the long-term sustainability of the codebase.

To support broad adoption within the OGS user community, the library is deliberately integrated at key points of interest, such as the official OGS benchmarks, executable test cases, and further contexts where previously used libraries were employed.

## Features

The implemented features are covering (1) pre-processing, (2) setup and execution of simulations, and (3) post-processing.

Preprocessing (1) for OGS includes mesh creation, adaptation, conversion, as well as defining boundary conditions, source terms, and generating project files (OGS specific XML-Files). Building upon this, a `FEFLOW` converter (from `FEFLOW` models to OGS models) is integrated \[@Heinze2025\]. The converter uses the geometric and material data of FEFLOW models to generate OGS-suitable meshes and definitions for H, HT and HC processes.

Postprocessing (3) includes domain specific evaluation and visualization of simulation results, for temporal and spatial distribution analysis.
`OGSTools` helps to create detailed plots by defining sensible defaults and OGS-specific standards.
It offers functionalities for the comparison of numerical simulation results with experimental data or analytical solutions.
Just as preprocessing and analysis are essential for single simulations, tooling becomes critical for efficiently handling ensemble runs.
Ensemble runs enable evaluation of model robustness, parameter sensitivity, numerical behaviour, and computational efficiency, with spatial and temporal grid conversion currently supported.

A more complete list of examples covering a significant part of the feature set is found in the online documentation of OGSTools [^1].
Containers are provided for reproducibility, benefiting both developers and users (\[@Bilke2025\]).
Like `OpenGeoSys`, `OGSTools` is available on `PyPI` and `Conda`.

## Applications

`OGSTools` library is designed to aid users in the implementation of complex workflows and has been an integral part of several scientific projects utilizing them.

### AREHS

The AREHS-Project (effects of changing boundary conditions on the development of hydrogeological systems: numerical long-term modelling considering thermal–hydraulic–mechanical(–chemical) coupled effects) project \[@Kahnt2021\] is focused on modelling the effects of the glacial cycle on hydro-geological parameters in potential geological nuclear waste repositories in Germany.
\[@Zill2024\] and \[@Silbermann2025\] highlighted the importance of automated workflow to efficiently develop models to answer the scientific question and to ensure the reproducibility of the results.
This workflow covers all necessary steps from a structured layered model and geological parameters over the simulation with OGS to the resulting figures shown in \[@Zill2024\] and \[@Silbermann2025\].
It is composed as a Snakemake workflow and all material is available on \[@arehs2024\].

### OpenWorkflow

`OpenWorkFlow` \[@openworkflow2023\], is a project for an open-source, modular synthesis platform designed for safety assessment in the nuclear waste site selection procedure of Germany.
Automated workflows as a piece of the planned scientific computational basis for investigating repository-induced physical and chemical processes in different geological setting are essential for transparent and reproducible simulation results.
OGS together with `OGSTools` has been used in a study of thermal repository dimensioning - named `ThEDi`. `ThEDi` focuses on determining the optimal packing of disposal containers in a repository to ensure temperature limits are not exceeded.
The fully automated workflow generates the simulation models based on geometric and material data, runs and analyses the simulations.
For scalability and parallelization the workflow is embedded optionally within the workflow management `Snakemake`.
The workflow components are implemented reusing `OGSTools` functionalities.

### OpenGeoSys benchmarks

The OGS benchmark gallery is a collection of web documents (mostly generated from `Jupyter Notebooks`) that demonstrate, how users can set up, adjust, execute, and analyse simulations.
They can be downloaded, executed, and adapted in an interactive environment for further exploration.
With `OGSTools` code complexity and code duplication has been reduced, and it allows especially inexperienced users to focus on the important part of the notebook.

The code transformation of the Mandel-Cryer effect benchmark, based on a simplified merge request[^2], demonstrates that using predefined plotting utilities reduces technical overhead.

Sections of the benchmark without OGSTools :

```python
# Mandel-Cryer effect benchmark
# Load the result
pvdfile = vtuIO.PVDIO("results_MandelCryerStaggered.pvd", dim=3)
fields = pvdfile.get_point_field_names()
time = pvdfile.timesteps

# Define observation points
observation_points = {"center": (0, 0, 0)}
pressure = pvdfile.read_time_series("pressure", observation_points)

# Plot pressure over time
fig, ax1 = plt.subplots(figsize=(10, 8))
ax1.plot(time, pressure["center"], color="tab:orange")
ax1.set_ylabel("Pressure (Pa)", fontsize=20)
ax1.set_xlabel("Time (s)", fontsize=20)
ax1.set_xlim(0, t_end)
ax1.set_ylim(0, 1500)
ax1.grid()
fig.supxlabel("Pressure at center of sphere")
plt.show()
```

By abstracting away lower-level plotting code, users can focus more directly on the physical interpretation of the benchmark results. Sections of the benchmark with OGSTools:

```python
# Mandel-Cryer effect benchmark without OGSTools
import ogstools as ot

ms = ot.MeshSeries("results_MandelCryerStaggered.pvd")

# Plot pressure over time
center_point = [0, 0, 0]
fig = ms.plot_probe(center_point, ot.variables.pressure, labels=["Center"])
```

### OGS-GIScape

`OGS-GIScape` is a `Snakemake`-based workflow for creating, simulating and analysing numerical groundwater models (NGM). OGS-GIScape enables scientists to investigate complex environmental models or conduct scenario analyses to study the groundwater flow and the associated environmental impact due to changes in groundwater resources.
Furthermore, the outcome of the models could be used for the management of groundwater resources.

An important part of the NGM creation is the geometric model (mesh).
It is build using geographic information system (GIS) tools at the landscape scale and combining various meshing tools.
The workflow also comprises the parametrization of the geometric model with physical parameters as well as defining boundary conditions, for instance groundwater recharge at the top of the computational domain or the integration of rivers.
For these workflow steps it is mainly necessary to change parts of the OGS project file which is done with `OGSTools`.

## Acknowledgements

This work has been supported by multiple funding sources, including `AREHS` under grant 4719F10402 by `Bundesamt für die Sicherheit der nuklearen Entsorgung`(BASE), and `OpenWorkflow` under grant STAFuE-21-05-Klei by `Bundesgesellschaft für Endlagerung (BGE)`.
The authors also acknowledge ongoing support from `SUTOGS` (Streamlining Usability and Testing of OpenGeoSys) under (grant \[Grant Number\]) by `Deutsche Forschungsgemeinschaft` (DFG)

[^1]: https://ogstools.opengeosys.org
[^2]: https://gitlab.opengeosys.org/ogs/ogs/-/merge_requests/5171
