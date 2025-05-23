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

OGSTools (OpenGeoSys Tools) is an open source (3-Clause BSD) Python library for streamlined usage of OpenGeoSys 6 (OGS), an also open-source software \[@ogs:6.5.4\] for simulating thermo-hydro-mechanical-chemical (THMC) processes in porous and fractured media \[@kolditz2012opengeosys\]. OGSTools provides an interface between OGS-specific data and well-established data structures of the Python ecosystem, as well as domain-specific solutions, examples, and tailored defaults for OpenGeoSys users and developers. The libraries' functionalities have proven to be main building blocks for complex automated workflows (including pre- and post-processing) designed to efficiently address scientific questions. For OGS users and developers, it serves as a framework for daily use.

# Statement of need

## Supporting modelers development efficiency

Modelers of OpenGeoSys alliteratively run simulations, analyse results, and refine their models with the goal to enhance the accuracy, efficiency and reliability of the simulation results. For efficiency, repetitive tasks of modelers development cycle should be formalized and run as computer programs. Python is a great choice for developer efficiency and scientific data analysis due to its simple syntax and vast ecosystem of libraries. The Python library introduced here is needed as central platform to collect common functionalities for building complex workflows, and enhancing the modelers' development cycle.

## Building block for complex workflows

A workflow is a structured sequence of tasks, that processes data and executes computations to achieve a specific goal \[@diercks\]. In our scientific researches, workflows need to integrate multiple steps—such as geological data preprocessing, ensemble simulation with OGS, domain specific analysis and visualization—into complex and fully automated and therefore reproducible sequences. Typically, one specific workflow is implemented to answer one specific scientific question. Workflow-based approach have been proofed to adhere to the FAIR principles \[@goble2020fair\], \[@Wilkinson_2025\]. In our approach we use existing workflow management software that covers domain independent parts like dependency graph description,computational efficiency, data management, execution control, multi-user collaboration, provenance and many more. Since we are building on the Python ecosystem, we have used Python-based workflow managers such as Snakemake \[@Snakemake\] and AiiDA \[@Huber_2020\].

The usage of workflow managers shifts the actual development to the workflow components. When common and frequently functionality found within workflow components are made reusable and provided as a Python library their reused in future workflows makes workflow development more efficient. The library should only contain functionalities directly related to the OGS core simulator and its specific input and output data, domain-specific definitions in geo-science, finite element modeling (FEM), and numerical computation. Our workflow components are then built of generic Python libraries, and our specific library and can be framed by any workflow management software. To ensure integration of the library's functionalities with the workflow management software, a list of functional and non-functional requirements (e.g., thread safety) imposed by all workflow management software we intend to use must be maintained and regularly validated through continuous testing.

## Centralization of Python-related development

Previously, our code base for Python-related tasks in OpenGeoSys was fragmented, with individual developers, modelers, and users creating their own separate solutions that often were not shared or standardized. This lack of centralization led to inefficiencies, inconsistent quality, and challenges in maintaining and extending the code. To address this, the all Python-related code needs to be centralized under the professional maintenance of the OGS core developer team. It ensures better collaboration, standardized practices and improved code quality. Further it enables the transfer of years of experience in maintaining the OpenGeoSys core (e.g., GitLab)  \[@Bilke2019\] to the Python side as well. For the centralized approach preceding work from MSH2VTU, OGS6PY, and VTUInterface \[@Buchwald\] and extracted functionalities from passed projects \[@AREHS\], \[@OPENWORKFLOW\]) needed to be adapted and integrated into a common library. To ensure broad adoption by the community—essential for the success of this centralized approach—the library must make features available so that all areas previously relying on the fragmented method are covered. This includes the OGS benchmarks, a set of executable Jupyter notebooks for demonstration of supported physical processes.

# Features

Features cover pre-processing, setup and execution of simulations and post-processing.

Preprocessing for OpenGeoSys includes mesh adaptation, conversion, refinement, and creation, as well as defining boundary conditions, source terms, and generating project files (OGS specific XML-Files). Building upon this, a FEFLOW converter (from FEFLOW models to OGS models) is integrated.

Postprocessing includes domain specific evaluation and visualization of simulation results, for temporal and spatial distribution analysis. It helps to create detailed plots by defining sensible matplotlib \[@matplotlib\] defaults and OGS-specific standards. OGSTools offers functionalities for the comparison of numerical simulation results with experimental data or analytical solutions.

Beyond single simulation preparation and analysis, multiple simulation analyses are present for evaluating computational and numerical efficiency and spatial grid conversion. For the later, OGSTools generates detailed convergence metrics according to \[@NASA\] based on simulations with progressively finer initial mesh discretizations.

A more complete list of examples covering a significant part of the features set is found in the online documentation of OGSTools\[Release DOCU\]. The up-to-date documentation \[@OGSTools_Latest_Docu\] is generated with Sphinx \[@Sphinx\], based on Jupyter \[@Jupyter\] notebooks. Further, the infrastructure includes dynamic testing with Pytest \[@Pytest\], and static code analysis MyPy \[@MyPy\] and a gitlab-instance. Containers are provided for reproducibility, benefiting both developers and users (\[@Bilke2\]). Like OpenGeoSys, OGSTools is available on PyPI \[@PyPI\] and Conda \[@CONDA\].

# Applications

OGSTools has been integral part of complex workflows. OGSTools is designed to support the implementation of such workflow components where OpenGeoSys in involved.

## AREHS

The AREHS-Project (effects of changing boundary conditions on the development of hydrogeological systems: numerical long-term modelling considering thermal–hydraulic–mechanical(–chemical) coupled effects) project \[@Kahnt\] focused on modeling the effects of the glacial cycle on hydro-geological parameters in potential geological nuclear waste repositories in Germany. \[@Kahnt\], \[@Zill\] and \[@Silbermann\] highlighted the importance of an automated workflows to efficiently develop models to answer the scientific question and to ensure the replicability of the results. The workflow covers all necessary steps from GOCAD \[@GOCAD_compoany\] structured layered model and geological parameters over the simulation with OpenGeoSys to the resulting figures shown in \[@Zill\] and \[@Silbermann\]. The workflow is composed as \[@SnakeMake\] workflow and all material available on \[@AREHS_Zenodo\] and documented on \[@AREHS_webdocu\].

## OpenWorkflow

OpenWorkFlow \[@Kolditz\], is a project for an open-source, modular synthesis platform designed for safety assessment in the nuclear waste site selection procedure of Germany.  Automated workflows as a piece of the planned scientific computational basis for investigating repository-induced physical and chemical processes in different geological setting are essential for transparent and replicable simulation results. OGS together with OGSTools has been used in a study of thermal repository dimensioning (ThEDi - "Thermische Endlager Dimensionierung"), that focuses on determining the optimal packing of disposal containers in a repository to ensure temperature limits are not exceeded. The fully automated workflow generates the simulation models based on geometric and material data, runs and analysis the simulations. For scalability and parallelization the workflow is embedded optionally within the workflow management Snakemake. The workflow components are implemented reusing OGSTools functionality.

## OpenGeoSys benchmarks

The OpenGeoSys benchmarks are a collection of web documents (mostly generated from Jupyter Notebooks) that demonstrate, how users can set up, adjust, execute, and analyse simulations. They can be downloaded and executed and adapted in an interactive environment for further exploration. With OGSTools code complexity and code duplication could be removed, and it allows especially inexperienced user to focus on the important part of the notebook.

# Acknowledgements

This work has been supported by multiple funding sources, including AREHS under grant \[Grant Number\], \[Other Funding Agency Name\] (grant \[Grant Number\]), and OpenWorkflow \[Additional Funding Agency Name\] (grant \[Grant Number\]). ExaESM. The authors also acknowledge ongoing support from SUTOGS (grant \[Grant Number\]).
