# SPDX-FileCopyrightText: Copyright (c) OpenGeoSys Community (opengeosys.org)
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

from lxml import etree as ET

from ogstools.ogs6py import build_tree


class ProcessVars(build_tree.BuildTree):
    """
    Managing the process variables section in the project file.
    """

    def __init__(self, tree: ET.ElementTree) -> None:
        self.tree = tree
        self.root = self.tree.getroot()
        self.pvs = self.populate_tree(
            self.root, "process_variables", overwrite=True
        )

    def set_ic(
        self,
        process_variable_name: str,
        components: int | str,
        order: int | str,
        initial_condition: str,
        **kwargs: Any,
    ) -> None:
        """
        Set initial conditions.

        For more details of the available arguments see
        https://doxygen.opengeosys.org/stable/d8/db8/ogs_file_param__prj__process_variables__process_variable.html

        :param process_variable_name:   name of the process variable
        :param components:              number of components
        :param order:                   polynomial order of shape functions
        :param initial_condition:       name of parameter holding the values of
                                        the initial condition
        """
        self._convertargs(kwargs)
        pv = self.populate_tree(self.pvs, "process_variable")
        self.populate_tree(pv, "name", text=process_variable_name)
        self.populate_tree(pv, "components", text=components)
        self.populate_tree(pv, "order", text=order)
        self.populate_tree(pv, "initial_condition", text=initial_condition)

        for tag, value in kwargs.items():
            self.populate_tree(pv, tag, value)

    def _find_process_var(self, process_variable_name: str) -> ET.Element:
        pv = self.root.find(
            f"./process_variables/process_variable[name='{process_variable_name}']"
        )
        if pv is None:
            msg = "You need to set initial condition for that process variable first."
            raise KeyError(msg)
        return pv

    def _check_mesh_or_geometry(self, **kwargs: Any) -> None:
        if "geometrical_set" not in kwargs and "mesh" not in kwargs:
            msg = "Please provide either a geometrical set or a mesh."
            raise KeyError(msg)
        if "geometrical_set" in kwargs and "geometry" not in kwargs:
            msg = "You also need to provide a geometry."
            raise KeyError(msg)

    def add_bc(
        self, process_variable_name: str, type: str, **kwargs: Any
    ) -> None:
        """
        Adds a boundary condition.

        Depending on the type of boundary condition different additional
        arguments need to be provided. For reference see
        https://doxygen.opengeosys.org/stable/d1/de4/ogs_file_param__prj__process_variables__process_variable__boundary_conditions__boundary_condition.html

        :param process_variable_name:   name of the process variable
        :param type:                    type of the boundary condition
        """
        self._convertargs(kwargs)
        pv = self._find_process_var(process_variable_name)
        self._check_mesh_or_geometry(**kwargs)

        bcs_element = self.find_or_populate(pv, "boundary_conditions")
        bc_element = self.populate_tree(bcs_element, "boundary_condition")
        self.populate_tree(bc_element, "type", type)
        for tag, value in kwargs.items():
            self.populate_tree(bc_element, tag, value)

    def add_st(
        self, process_variable_name: str, type: str, **kwargs: Any
    ) -> None:
        """
        Add a source term.

        Depending on the type of source term different additional
        arguments need to be provided. For reference see
        https://doxygen.opengeosys.org/stable/dc/dfc/ogs_file_param__prj__process_variables__process_variable__source_terms__source_term.html

        :param process_variable_name:   name of the process variable
        :param type:                    type of the source term
        """
        self._convertargs(kwargs)
        pv = self._find_process_var(process_variable_name)
        self._check_mesh_or_geometry(**kwargs)

        source_terms = self.find_or_populate(pv, "source_terms")
        source_term = self.populate_tree(source_terms, "source_term")
        self.populate_tree(source_term, "type", text=type)
        for tag, value in kwargs.items():
            self.populate_tree(source_term, tag, value)

    def deactivate_subdomain(
        self, process_variable_name: str, **kwargs: Any
    ) -> None:
        """
        Deactivate subdomain/s for a specific process variable.

        https://doxygen.opengeosys.org/stable/da/d70/ogs_file_param__prj__process_variables__process_variable__deactivated_subdomains__deactivated_subdomain.html

        :param process_variable_name:   name of the process variable
        """
        self._convertargs(kwargs)
        pv = self._find_process_var(process_variable_name)
        deact_subs = self.find_or_populate(pv, "deactivated_subdomains")
        deact_sub = self.populate_tree(deact_subs, tag="deactivated_subdomain")
        for tag, value in kwargs.items():
            self.populate_tree(deact_sub, tag, value)
