# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

from contextlib import suppress

from lxml import etree as ET

try:
    from IPython.display import Markdown, display

    verbose = True
except ImportError:
    verbose = False


class Display:
    """Helper class to create a nested dictionary
    representing the xml structure.
    """

    def __init__(self, tree: ET.ElementTree):
        if verbose is True:
            display(Markdown("## OpenGeoSys project file"))
            display(Markdown("### Main Info"))
            with suppress(AttributeError):
                display(
                    Markdown(
                        f"**Process name:** {tree.find('./processes/process/name').text}"
                    )
                )
            with suppress(AttributeError):
                display(
                    Markdown(
                        f"**Process type:** {tree.find('./processes/process/type').text}"
                    )
                )
            with suppress(AttributeError):
                t_end = float(
                    tree.find(
                        "./time_loop/processes/process/time_stepping/t_end"
                    ).text
                )
                t_init = float(
                    tree.find(
                        "./time_loop/processes/process/time_stepping/t_initial"
                    ).text
                )
                display(Markdown(f"**Simulation time:** {t_end-t_init}"))
            with suppress(AttributeError):
                proc_vars = tree.findall("./process_variables/process_variable")
                with suppress(AttributeError):
                    display(
                        Markdown(
                            f"**Output prefix:** {tree.find('./time_loop/output/prefix').text}"
                        )
                    )
                display(Markdown("### Boundary conditions"))
                for var in proc_vars:
                    proc_var_entries = var.getchildren()
                    for entry in proc_var_entries:
                        if entry.tag == "name":
                            display(
                                Markdown(f"**Process Variable:** {entry.text}")
                            )
                        if entry.tag == "initial_condition":
                            display(
                                Markdown(
                                    f" - **Initial Condition:** {entry.text}"
                                )
                            )
                        if entry.tag == "order":
                            display(Markdown(f" - **Order:** {entry.text}"))
                        if entry.tag == "boundary_conditions":
                            bcs = entry.getchildren()
                            for bc in bcs:
                                bc_entries = bc.getchildren()
                                for subentry in bc_entries:
                                    if subentry.tag == "type":
                                        display(
                                            Markdown(
                                                f" - **Type:** {subentry.text}"
                                            )
                                        )
                                    if subentry.tag == "mesh":
                                        display(
                                            Markdown(
                                                f" - **Mesh:** {subentry.text}"
                                            )
                                        )
                                    if subentry.tag == "geometrical_set":
                                        display(
                                            Markdown(
                                                f" - **Geometrical Set:** {subentry.text}"
                                            )
                                        )
                                    if subentry.tag == "geometry":
                                        display(
                                            Markdown(
                                                f" - **Geometry:** {subentry.text}"
                                            )
                                        )
