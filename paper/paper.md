______________________________________________________________________

title: OGSTools
tags:

- Python
- OpenGeoSys
- Workflows
- finite element method
- geosciences
- Computer simulation
- preprocessing
- modeling
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
  affiliation: 1,2
- name: Julian Heinze
  orcid: 0009-0004-3449-8852
  equal-contrib: true
  affiliation: 1
- name: Lars Bilke
  orcid: 0000-0001-8986-2413
  equal-contrib: true
  affiliation: 1
- name: Max Jäschke
  equal-contrib: true
  affiliation: 3
- name: Feliks Kiszkurno
  orcid: 0000-0003-3304-4838
  equal-contrib: true
  affiliation: 1
- name: Dominik Kern
  orcid: 0000-0002-1958-2982
  equal-contrib: true
  affiliation: 2
- name: Olaf Kolditz
  orcid: 0000-0002-8098-4905
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
  date: 17 March 2025
  bibliography: paper.bib

______________________________________________________________________

# Summary

OGSTools (OpenGeoSys Tools) is an open source (3-Clause BSD) Python library for streamlined usage of OpenGeoSys 6 (OGS), also an open-source software \[@ogs:6.5.4\] for simulating thermo-hydro-mechanical-chemical (THMC) processes in porous and fractured media \[@kolditz2012opengeosys\]. OGSTools provides an interface between OGS-specific data and well-established data structures of the Python ecosystem, as well as domain-specific solutions, examples, and tailored defaults for OGS users and developers. The libraries' functionalities have been proven to be useful in complex automated workflows (including pre- and post-processing), the OGS benchmark gallery, the OGS test-suite, and in automating repetitive tasks in the model development cycle.

# Statement of need

## Development efficiency

Modelers of OGS iteratively run simulations, analyse results, and refine their models with the goal to enhance the accuracy, efficiency and reliability of the simulation results. To improve efficiency, repetitive tasks in the model development cycle should be formalized and automated through computer programs. Python is a great choice for developer efficiency and scientific data analysis due to its simple syntax and vast ecosystem of libraries. The Python library introduced here is needed as a central platform to collect and improve common functionalities.

## Complex workflows

A workflow is a structured sequence of tasks, that processes data and executes computations to achieve a specific goal \[@diercks\]. In our scientific research, workflows need to integrate multiple steps—such as geological data preprocessing, ensemble simulations with OGS, domain specific analysis and visualization—into complex and fully automated and therefore reproducible sequences. Typically, one specific workflow is implemented to answer one specific scientific question. Workflow-based approaches have been proofed to adhere to the FAIR principles \[@goble2020fair\], \[@Wilkinson_2025\]. In our approach we use existing workflow management software that covers domain independent parts like dependency graph description, computational efficiency, data management, execution control, multi-user collaboration and data provenance \[@Bilke2025\]. Since we are building on the Python ecosystem, using Python-based workflow managers such as Snakemake \[@Snakemake\] and AiiDA \[@Huber_2020\] is a natural choice.

The usage of workflow managers shifts the actual development to the workflow components. Common and frequently used functionality found within workflow components are made reusable and provided as a Python library their repeated use in future workflows makes workflow development more efficient. The library focuses on functionalities directly related to (1) the OGS core simulator and its specific input and output data, (2) domain-specific definitions in geo-science, (3) finite element modeling (FEM), and (4) numerical computation. Our workflow components are then built of generic Python libraries, our library and workflow manager dedicated integration code. To ensure integration of the library's functionalities with the workflow management software, a list of functional and non-functional requirements (e.g., thread safety), imposed by all workflow management software we intend to use, are maintained and regularly validated through continuous testing.

Code example

## Test suite

## Centralization of Python-related development

Previously, our code base for Python-related tasks in OGS was fragmented, with components, that were often developed for specific use cases, with varying degrees of standardization and sharing. The lack of centralization led to inefficiencies, inconsistent quality, and challenges in maintaining and extending the code. With OGSTools, all Python-related code is now centralized under the professional maintenance of the OGS core developer team. It ensures better collaboration, standardized practices and improved code quality. Further it enables the transfer of years of experience in maintaining the OGS core, written in C++ \[GitLab\],  \[@Bilke2019\] to the pre- and post-processing code, written in Python. For the centralized approach, preceding work from MSH2VTU, OGS6PY, and VTUInterface \[@Buchwald\] and extracted functionalities from passed projects \[@AREHS\], \[@OPENWORKFLOW\]) have been adapted and integrated into OGSTools.

We recommend our users to setup a project specific (virtual) python environment or conda environment and freeze the dependencies. Otherwise these Python environments follow a rolling-release model, in which dependencies are updated continuously without centralized coordination, potentially leading to unexpected behavior or incompatibilities. To address \[\\The Need for a Versioned Data Analysis Software Environment\] OGSTools provides additionally a pinned environment, updated at least once per release. While reproducibility requires environments with pinned dependencies, OGSTools is additionally tested with the latest dependencies to receive early warning of breaking changes and support long-term sustainability of the codebase.

To support broad adoption within the OGS user community, the library is deliberately integrated at key points of interest, such as the official OGS benchmarks, executable test cases, and further contexts where previously used libraries were employed.

# Features

The implemented features are covering pre-processing, setup and execution of simulations and post-processing.

Preprocessing for OGS includes mesh adaptation, conversion, refinement, and creation, as well as defining boundary conditions, source terms, and generating project files (OGS specific XML-Files). Building upon this, a FEFLOW converter (from FEFLOW models to OGS models) is integrated.

Postprocessing includes domain specific evaluation and visualization of simulation results, for temporal and spatial distribution analysis. OGSTools helps to create detailed plots by defining sensible matplotlib \[@matplotlib\] defaults and OGS-specific standards. It offers functionalities for the comparison of numerical simulation results with experimental data or analytical solutions. Just as preprocessing and analysis are essential for single simulations,tooling becomes critical for efficiently handling ensemble runs. Ensemble runs enable evaluation of model robustness, parameter sensitivity, numerical behavior, and computational efficiency, with spatial and temporal grid conversion currently supported. For the latter, OGSTools generates detailed convergence metrics according to \[@NASA\] based on simulations with progressively finer initial mesh discretizations.

A more complete list of examples covering a significant part of the features set is found in the online documentation of OGSTools\[Release DOCU\]. The up-to-date documentation \[@OGSTools_Latest_Docu\] is generated with Sphinx \[@Sphinx\], based on Jupyter \[@Jupyter\] notebooks. Further, the infrastructure includes dynamic testing with Pytest \[@Pytest\], and static code analysis MyPy \[@MyPy\] and a gitlab-instance. Containers are provided for reproducibility, benefiting both developers and users (\[@Bilke2\]). Like OpenGeoSys, OGSTools is available on PyPI \[@PyPI\] and Conda \[@CONDA\].

# Applications

OGSTools has been integral part of complex workflows. OGSTools is designed to support the implementation of such workflow components where OGS in involved.

## AREHS

The AREHS-Project (effects of changing boundary conditions on the development of hydrogeological systems: numerical long-term modelling considering thermal–hydraulic–mechanical(–chemical) coupled effects) project \[@Kahnt\] focused on modeling the effects of the glacial cycle on hydro-geological parameters in potential geological nuclear waste repositories in Germany. \[@Kahnt\], \[@Zill\] and \[@Silbermann\] highlighted the importance of an automated workflows to efficiently develop models to answer the scientific question and to ensure the reproducibility of the results. This workflow covers all necessary steps from GOCAD \[@GOCAD_compoany\] structured layered model and geological parameters over the simulation with OGS to the resulting figures shown in \[@Zill\] and \[@Silbermann\]. It is composed as \[@SnakeMake\] workflow and all material available on \[@AREHS_Zenodo\] and documented on \[@AREHS_webdocu\].

## OpenWorkflow

OpenWorkFlow \[@Kolditz\], is a project for an open-source, modular synthesis platform designed for safety assessment in the nuclear waste site selection procedure of Germany.  Automated workflows as a piece of the planned scientific computational basis for investigating repository-induced physical and chemical processes in different geological setting are essential for transparent and reproducible simulation results. OGS together with OGSTools has been used in a study of thermal repository dimensioning (ThEDi - German: "Thermische Endlager Dimensionierung"), that focuses on determining the optimal packing of disposal containers in a repository to ensure temperature limits are not exceeded. The fully automated workflow generates the simulation models based on geometric and material data, runs and analyses the simulations. For scalability and parallelization the workflow is embedded optionally within the workflow management Snakemake. The workflow components are implemented reusing OGSTools functionalities.

## OpenGeoSys benchmarks

The OGS benchmarks are a collection of web documents (mostly generated from Jupyter Notebooks) that demonstrate, how users can set up, adjust, execute, and analyse simulations. They can be downloaded, executed, and adapted in an interactive environment for further exploration. With OGSTools code complexity and code duplication could be reduced, and it allows especially inexperienced users to focus on the important part of the notebook.

# Acknowledgements

This work has been supported by multiple funding sources, including AREHS under grant \[Grant Number\], \[Other Funding Agency Name\] (grant \[Grant Number\]), and OpenWorkflow \[Additional Funding Agency Name\] (grant \[Grant Number\]). The authors also acknowledge ongoing support from SUTOGS (grant \[Grant Number\]).
