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
- name: Norbert Grunwald
  orcid: 0000-0002-5264-2246
  equal-contrib: true
  affiliation: 1
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

`OGSTools` (`OpenGeoSys` Tools) is a Python library for pre- and post-processing of `OpenGeoSys 6` (OGS) - a software package for simulating thermo-hydro-mechanical-chemical (THMC) processes in porous and fractured media \[@bilke_2025_14672997\] \[@kolditz2012opengeosys\].
`OGSTools` \[@ogstools2025\] provides an interface between OGS-specific data and well-established data structures of the Python ecosystem, as well as domain-specific solutions, examples, and sensible defaults for OGS users and developers. By connecting OGS to the ecosystem of Python, the entry threshold to the OGS platform is lowered for users.
The library's functionalities are designed to be used in the OGS benchmark gallery, the OGS test suite, and for automating repetitive tasks in the model development cycle — from simple daily tasks to complex automated workflows.

## Statement of need

### Development efficiency

Modellers of OGS iteratively run simulations, analyse results, and refine their models with the goal to improve the accuracy, efficiency and reliability of the simulation results.
To improve efficiency, repetitive steps in the model development cycle should be formalised. Python was chosen as the formalisation language because it matches the existing expertise of the user base.
The Python library introduced here serves as a central platform to collect and improve common functionalities needed by modellers of OGS.

### Complex workflows

A workflow is a structured sequence of steps, that processes data and executes computations to achieve a specific goal \[@diercks2022workflows\].
In our scientific research, workflows need to integrate multiple steps—such as geological data preprocessing, ensemble simulations with OGS, domain-specific analysis and visualization—into complex, fully automated, and therefore reproducible sequences.
Typically, one specific workflow is implemented to answer one specific scientific question.
Workflow-based approaches have been proven to adhere to the `FAIR principles` \[@goble2020fair\], \[@Wilkinson_2025\]. The typical approach is to use existing workflow management software that covers domain-independent parts like dependency graph description, computational efficiency, data management, execution control, multi-user collaboration and data provenance \[@Bilke2025\].
Building on the Python ecosystem, our goal is an integrated solution in which all components, including the Python-based workflow managers like `Snakemake` \[@Köster2012\] and `AiiDA` \[@Huber2020\], function together.
Common and frequently used functionality found within workflow components is made reusable and provided in this Python library.
It focuses on functionalities directly related to (1) the OGS core simulator and its specific input and output data, (2) domain-specific definitions in geo-science, (3) finite element modelling (FEM), and (4) numerical computation.
The workflow components are constructed from generic Python libraries, OGSTools, and integration code for the respective workflow manager chosen.

### Test suite

`OGSTools` provides functionality for (1) setting up test environments, (2) executing OGS under specified conditions, (3) evaluating OGS results against defined test criteria, and (4) monitoring the testing process.

### Educational Jupyter notebooks

OGS is already being used in academic courses and teaching environments. With Jupyter Notebooks, students can explore interactive learning environments where they directly modify parameters, material laws, and other influencing factors, and instantly visualize the outcomes. `OGSTools` serves as an interface between OGS and Jupyter Notebooks. It supports the creation of input data—such as easily configurable meshes or ready-to-use project files.

### Centralization of Python-related development

Previously, the code base for Python-related tasks in OGS was fragmented, with components often developed for specific use cases and varying degrees of standardization.
The lack of centralization led to inefficiencies, inconsistent quality, and challenges in maintaining and extending the code.
With `OGSTools`, reusable Python code is now centralized under the professional maintenance of the core developer team of OGS.
Further, it enables the transfer of years of experience in maintaining the OGS core \[@Bilke2019\] to the pre- and post-processing code.
For the centralized approach, preceding work on `msh2vtu` \[@msh2vtu\], `ogs6py and VTUInterface` \[@Buchwald2021\] and extracted functionalities from the projects (1) `AREHS` \[@arehs2024\], and (2) `OpenWorkFlow - Synthesis Platform` \[@openworkflow2023\] have been adapted and integrated into `OGSTools`.

To address `The Need for a Versioned Data Analysis Software Environment` \[@Blomer2014\] `OGSTools` additionally provides a pinned environment, updated at least once per release.
While reproducibility requires environments with pinned dependencies, `OGSTools` is additionally tested with the latest dependencies, to receive early warnings of breaking changes and to support the long-term sustainability of the codebase. To support broad adoption within the OGS user community, the library is deliberately integrated at key points of interest, such as the official OGS benchmarks, executable test cases, and further contexts where previously used libraries were employed.

## Features

The implemented features are covering pre-processing, setup and execution of simulations, and post-processing.

Pre-processing for OGS includes mesh creation, adaptation, conversion, as well as defining boundary conditions, source terms, and generating project files (OGS-specific XML files). OGSTools further provides a material management component that allows process-specific material definitions to be assembled from structured YAML sources and translated into OGS-compatible project file entries. This allows a consistent, database-like handling of material parameters across workflows, test cases, and educational examples, while separating physical model definitions from project file syntax. In addition, a `FEFLOW` converter (from `FEFLOW` models to OGS models) is integrated \[@Heinze2025\]. The converter uses the geometric and material data of FEFLOW models to generate OGS-suitable meshes and definitions for H, HT and HC processes.

The simulation execution part covers running simulations with the `OGS` core the via command line and Python-based co-simulation interfaces. Runtime features include monitoring, interactive stepping, and access to intermediate results for in-simulation analysis.

Post-processing includes domain-specific evaluation and visualization of simulation results, for temporal and spatial distribution analysis.
`OGSTools` helps to create detailed plots by defining sensible defaults and OGS-specific standards.
It offers functionalities for the comparison of numerical simulation results with experimental data or analytical solutions.
Just as preprocessing and analysis are essential for single simulations, tooling becomes critical for efficiently handling ensemble runs.
Ensemble runs enable evaluation of model robustness, parameter sensitivity, numerical behaviour, and computational efficiency, with spatial and temporal grid conversion currently supported.

A more complete list of examples covering a significant part of the feature set is found in the online documentation of OGSTools [^1].
Containers are provided for reproducibility, benefiting both developers and users (\[@Bilke2025\]).
Like `OpenGeoSys`, `OGSTools` is available on `PyPI` and `Conda`.

## Applications

### Workflows

The AREHS-Project (effects of changing boundary conditions on the development of hydrogeological systems: numerical long-term modelling considering thermal–hydraulic–mechanical(–chemical) coupled effects) \[@Kahnt2021\] is focused on modelling the effects of the glacial cycle on hydro-geological parameters in potential geological nuclear waste repositories in Germany.
\[@Zill2024\] and \[@Silbermann2025\] highlighted the importance of an automated workflow to efficiently develop models to answer the scientific question and to ensure the reproducibility of the results. For reproducibility all material is available at \[@arehs2024\].
`OpenWorkFlow` \[@openworkflow2023\] is a project for an open-source, modular synthesis platform designed for safety assessment in the nuclear waste site selection procedure of Germany. `ThEDi` is a study, that focuses on determining the optimal packing of disposal containers in a repository to ensure temperature limits are not exceeded.
`OGS-GIScape` is a workflow for creating, simulating and analysing numerical groundwater models. OGS-GIScape helps scientists to investigate complex environmental models or conduct scenario analyses to study the groundwater flow and the associated environmental impact due to changes in groundwater resources. The outcome of the models could be used for the management of groundwater resources.
For scalability and parallelization all mentioned workflows use the workflow management software `Snakemake`. The rules are implemented using `OGSTools`.

### OpenGeoSys benchmarks

The OGS benchmark gallery is a collection of web documents (mostly generated from `Jupyter Notebooks`) that demonstrate, how users can set up, adjust, execute, and analyse simulations.
They can be downloaded, executed, and adapted in an interactive environment for further exploration. With `OGSTools`, code complexity and code duplication has been reduced, and it allows especially inexperienced users to focus on the important part of the notebook.

## Example



## Acknowledgements

This work has been supported by multiple funding sources, including `AREHS` under grant 4719F10402 by `Bundesamt für die Sicherheit der nuklearen Entsorgung (BASE)`, and `OpenWorkflow` under grant STAFuE-21-05-Klei by `Bundesgesellschaft für Endlagerung (BGE)`.
The authors also acknowledge ongoing support from `SUTOGS` (Streamlining Usability and Testing of OpenGeoSys) under (grant \[Grant Number\]) by `Deutsche Forschungsgemeinschaft` (DFG)

[^1]: https://ogstools.opengeosys.org