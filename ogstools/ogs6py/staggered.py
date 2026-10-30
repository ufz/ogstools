# SPDX-FileCopyrightText: Copyright (c) OpenGeoSys Community (opengeosys.org)
# SPDX-License-Identifier: BSD-3-Clause

"""Conversion of a monolithic project into one using the staggered scheme."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from lxml import etree as ET

if TYPE_CHECKING:
    from ogstools.ogs6py.project import Project

# The processes that can be solved with the staggered scheme. Each process type
# maps to the tags of its process variables in the order of the staggered
# process ids, which is also their order in a monolithic per-component
# convergence criterion, and to whether the process reads <coupling_scheme>
# with the coupling scheme parser of HYDRO_MECHANICS, i.e. as a subtree with a
# <type>, rather than as a plain string.
STAGGERED_PROCESSES: dict[str, tuple[tuple[str, ...], bool]] = {
    "HT": (("temperature", "pressure"), False),
    "HYDRO_MECHANICS": (("pressure", "displacement"), True),
    "RICHARDS_MECHANICS": (("pressure", "displacement"), False),
    "THERMO_HYDRO_MECHANICS": (
        ("temperature", "pressure", "displacement"),
        True,
    ),
    "THERMO_MECHANICS": (("temperature", "displacement"), False),
}

_PER_COMPONENT_TYPES = {
    "PerComponentDeltaX": "DeltaX",
    "PerComponentResidual": "Residual",
}
_SCALAR_TYPES = ("DeltaX", "Residual")


def convert_to_staggered(
    project: Project,
    coupling_criteria: Sequence[Mapping[str, Any]],
    max_iter: int,
    coupling_scheme: str | Mapping[str, Any] | None = None,
    linear_solver_settings: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    """Convert a project using the monolithic scheme to the staggered scheme.

    The project must contain one process, solved by one time loop process.
    That time loop process is replaced by one time loop process per process
    variable, in the order of the staggered process ids (e.g. temperature,
    pressure, displacement for THERMO_HYDRO_MECHANICS). Each of them is a copy
    of the monolithic one with its own copy of the nonlinear and of the
    linear solver, whose names get the tag of the process variable as suffix
    (e.g. ``basic_newton_temperature``).

    The convergence criterion of each sub-problem is derived from the
    monolithic one: a per-component criterion is split by the numbers of
    components of the process variables, and a scalar criterion is copied.

    :param project:             The project, which is modified in place.
    :param coupling_criteria:   One convergence criterion of the coupling
                                iteration per staggered process id, each given
                                by the keyword arguments of
                                :meth:`~ogstools.ogs6py.timeloop.TimeLoop.add_global_process_coupling`,
                                e.g. ``{"convergence_type": "DeltaX",
                                "norm_type": "NORM2", "reltol": 1e-8}``.
    :param max_iter:            Maximum number of coupling iterations per time
                                step.
    :param coupling_scheme:     The ``<coupling_scheme>`` of the process: a
                                string is written as its text, a mapping as
                                its children, e.g. ``{"type": "staggered",
                                "fixed_stress_stabilization_parameter": 0.6}``.
                                By default, the form the process reads in the
                                current OGS release is used.
    :param linear_solver_settings:  Optionally, one mapping per staggered
                                process id from an XPath relative to the
                                ``<linear_solver>`` element to a new text, e.g.
                                ``{"eigen/error_tolerance": 1e-16}``.
    """
    root = project._get_root()

    processes = root.findall("./processes/process")
    if len(processes) != 1:
        msg = f"Expected one process, found {len(processes)}."
        raise ValueError(msg)
    process = processes[0]
    process_type = process.findtext("type")
    if process_type not in STAGGERED_PROCESSES:
        msg = (
            f"The staggered scheme is not supported for process type "
            f"{process_type}. Supported: {', '.join(STAGGERED_PROCESSES)}."
        )
        raise ValueError(msg)
    if process.find("coupling_scheme") is not None:
        msg = "The process already has a <coupling_scheme>."
        raise ValueError(msg)
    variable_tags, hm_coupling_scheme = STAGGERED_PROCESSES[process_type]

    time_loop_processes = root.findall("./time_loop/processes/process")
    if len(time_loop_processes) != 1:
        msg = (
            "Expected one process in the time loop, found "
            f"{len(time_loop_processes)}."
        )
        raise ValueError(msg)
    monolithic_process = time_loop_processes[0]
    global_process_coupling = root.find("./time_loop/global_process_coupling")
    if global_process_coupling is not None and len(global_process_coupling):
        msg = "The time loop already has a <global_process_coupling>."
        raise ValueError(msg)

    numbers_of_components = [
        _number_of_components(root, process, tag) for tag in variable_tags
    ]
    _check_length("coupling_criteria", coupling_criteria, len(variable_tags))
    if linear_solver_settings is None:
        linear_solver_settings = [{}] * len(variable_tags)
    _check_length(
        "linear_solver_settings", linear_solver_settings, len(variable_tags)
    )

    convergence_criteria = _split_convergence_criterion(
        _find(monolithic_process, "convergence_criterion"),
        numbers_of_components,
    )

    nonlinear_solver_name = _findtext(monolithic_process, "nonlinear_solver")
    nonlinear_solvers = _find(root, "./nonlinear_solvers")
    nonlinear_solver = _find(
        nonlinear_solvers, f"nonlinear_solver[name='{nonlinear_solver_name}']"
    )
    linear_solver_name = _findtext(nonlinear_solver, "linear_solver")
    linear_solvers = _find(root, "./linear_solvers")
    linear_solver = _find(
        linear_solvers, f"linear_solver[name='{linear_solver_name}']"
    )

    _add_coupling_scheme(process, coupling_scheme, hm_coupling_scheme)

    time_loop_processes_element = _find(root, "./time_loop/processes")
    position = time_loop_processes_element.index(monolithic_process)
    time_loop_processes_element.remove(monolithic_process)
    nonlinear_solvers.remove(nonlinear_solver)
    if (
        nonlinear_solvers.find(
            f"nonlinear_solver[linear_solver='{linear_solver_name}']"
        )
        is None
    ):
        linear_solvers.remove(linear_solver)

    for index, (tag, convergence_criterion, settings) in enumerate(
        zip(
            variable_tags,
            convergence_criteria,
            linear_solver_settings,
            strict=True,
        )
    ):
        sub_nonlinear_solver_name = f"{nonlinear_solver_name}_{tag}"
        sub_linear_solver_name = f"{linear_solver_name}_{tag}"

        sub_process = copy.deepcopy(monolithic_process)
        _find(sub_process, "nonlinear_solver").text = sub_nonlinear_solver_name
        sub_process.replace(
            _find(sub_process, "convergence_criterion"), convergence_criterion
        )
        time_loop_processes_element.insert(position + index, sub_process)

        sub_nonlinear_solver = copy.deepcopy(nonlinear_solver)
        _find(sub_nonlinear_solver, "name").text = sub_nonlinear_solver_name
        _find(sub_nonlinear_solver, "linear_solver").text = (
            sub_linear_solver_name
        )
        nonlinear_solvers.append(sub_nonlinear_solver)

        sub_linear_solver = copy.deepcopy(linear_solver)
        _find(sub_linear_solver, "name").text = sub_linear_solver_name
        for xpath, value in settings.items():
            _find(sub_linear_solver, xpath).text = str(value)
        linear_solvers.append(sub_linear_solver)

    for index, criterion in enumerate(coupling_criteria):
        if index == 0:
            project.time_loop.add_global_process_coupling(
                max_iter=max_iter, **criterion
            )
        else:
            project.time_loop.add_global_process_coupling(**criterion)


def _find(element: ET._Element, path: str) -> ET._Element:
    found = element.find(path)
    if found is None:
        msg = f"<{element.tag!s}> has no {path}."
        raise ValueError(msg)
    return found


def _findtext(element: ET._Element, path: str) -> str:
    text = element.findtext(path)
    if text is None:
        msg = f"<{element.tag!s}> has no {path}."
        raise ValueError(msg)
    return text.strip()


def _check_length(name: str, values: Sequence[Any], expected: int) -> None:
    if len(values) != expected:
        msg = (
            f"{name} needs one entry per staggered process id, i.e. "
            f"{expected}, but has {len(values)}."
        )
        raise ValueError(msg)


def _number_of_components(
    root: ET._Element, process: ET._Element, tag: str
) -> int:
    variable_name = _findtext(process, f"process_variables/{tag}")
    return int(
        _findtext(
            root,
            f"./process_variables/process_variable[name='{variable_name}']"
            "/components",
        )
    )


def _split_convergence_criterion(
    criterion: ET._Element, numbers_of_components: Sequence[int]
) -> list[ET._Element]:
    criterion_type = _findtext(criterion, "type")
    if criterion_type in _SCALAR_TYPES:
        return [copy.deepcopy(criterion) for _ in numbers_of_components]
    if criterion_type not in _PER_COMPONENT_TYPES:
        msg = f"Cannot split a convergence criterion of type {criterion_type}."
        raise ValueError(msg)

    tolerances = {}
    for tag in ("abstols", "reltols"):
        text = criterion.findtext(tag)
        if text is None:
            continue
        values = text.split()
        if len(values) != sum(numbers_of_components):
            msg = (
                f"<{tag}> has {len(values)} values, but the process "
                f"variables have {sum(numbers_of_components)} components."
            )
            raise ValueError(msg)
        tolerances[tag] = values

    criteria = []
    first = 0
    for number_of_components in numbers_of_components:
        last = first + number_of_components
        sub_criterion = copy.deepcopy(criterion)
        for tag, values in tolerances.items():
            sub_criterion.remove(_find(sub_criterion, tag))
            if number_of_components == 1:
                ET.SubElement(sub_criterion, tag[:-1]).text = values[first]
            else:
                ET.SubElement(sub_criterion, tag).text = " ".join(
                    values[first:last]
                )
        if number_of_components == 1:
            _find(sub_criterion, "type").text = _PER_COMPONENT_TYPES[
                criterion_type
            ]
        criteria.append(sub_criterion)
        first = last
    return criteria


def _add_coupling_scheme(
    process: ET._Element,
    coupling_scheme: str | Mapping[str, Any] | None,
    hm_coupling_scheme: bool,
) -> None:
    if coupling_scheme is None:
        coupling_scheme = (
            {"type": "staggered"} if hm_coupling_scheme else "staggered"
        )
    element = ET.Element("coupling_scheme")
    if isinstance(coupling_scheme, str):
        element.text = coupling_scheme
    else:
        for tag, value in coupling_scheme.items():
            ET.SubElement(element, tag).text = str(value)
    # Insert it after <type>, where it is easy to find.
    process.insert(process.index(_find(process, "type")) + 1, element)
