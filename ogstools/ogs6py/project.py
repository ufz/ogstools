# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

"""
ogs6py is a python-API for the OpenGeoSys finite element software.
Its main functionalities include creating and altering OGS6 input files as well as executing OGS.

"""

import copy
import difflib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from lxml import etree as ET
from typing_extensions import Self

from ogstools.core.storage import StorageBase
from ogstools.materiallib.core.media import MediaSet
from ogstools.ogs6py import (
    curves,
    display,
    geo,
    linsolvers,
    local_coordinate_system,
    media,
    mesh,
    nonlinsolvers,
    parameters,
    processes,
    processvars,
    python_script,
    timeloop,
)
from ogstools.ogs6py.project_media_importer import _ProjectMediaImporter
from ogstools.ogs6py.properties import (
    Property,
    PropertySet,
    Value,
    _expand_tensors,
    _expand_van_genuchten,
    location_pointer,
    property_dict,
)


class Project(StorageBase):
    """Class for handling an OGS6 project.

    In this class everything for an OGS6 project can be specified.
    """

    def __init__(
        self,
        input_file: str | Path | None = None,
        output_file: str | Path | None = None,
        output_dir: str | Path = Path(),
        logfile: str | Path = "out.log",
        verbose: bool = False,
        xml_string: str | None = None,
        id: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Create a new Project instance.

        :param input_file:  Filename of the input project file
        :param output_file: Filename of the output project file
        :param output_dir:  Directory of the simulation output
        :param logfile:     Filename into which the log is written
        :param xml_string:  String containing the XML tree
        :param verbose:     If True, show verbose output

        Optional Keyword Arguments:
            - OMP_NUM_THREADS: int, sets the environment variable before OGS
                execution to restrict number of OMP Threads
            - OGS_ASM_THREADS: int, sets the environment variable before OGS
                execution to restrict number of OMP Threads
        """
        sys.setrecursionlimit(10000)
        super().__init__("Project", "", id=id)
        self.logfile = Path(logfile)
        self.process: None | subprocess.Popen = None
        self.runtime_start: float = 0.0
        self.runtime_end: float = 0.0
        self.tree: ET._ElementTree | None = None
        self.include_elements: list[ET._Element] = []
        self.include_files: list[Path] = []
        self.add_includes: list[dict[str, str]] = []
        self.output_dir = Path(output_dir)  # default -> current dir
        self.verbose = verbose
        self.threads: int | None = kwargs.get("OMP_NUM_THREADS")
        self.asm_threads: int = kwargs.get("OGS_ASM_THREADS", self.threads)
        self.input_file: Path | None = None
        self.folder: Path = Path()

        if output_file:
            # Legacy: output_file sets prjfile for write_input()
            self.prjfile = Path(output_file)
        elif input_file is not None:
            self.prjfile = Path(input_file)
            self._bind_to_path(input_file.parent)
        else:
            self.prjfile = self._next_target / "project.prj"

        if input_file is not None:
            input_file = Path(input_file)
            if input_file.is_file():
                self.input_file = input_file
                self.folder = input_file.parent
                _ = self._get_root()
                if self.verbose is True:
                    display.Display(self.tree)
            else:
                msg = f"Input project file {input_file} not found."
                raise FileNotFoundError(msg)
        else:
            self.input_file = None
            self.root = ET.Element("OpenGeoSysProject")
            parse = ET.XMLParser(remove_blank_text=True, huge_tree=True)
            tree_string = ET.tostring(self.root, pretty_print=True)
            self.tree = ET.ElementTree(ET.fromstring(tree_string, parser=parse))
        if xml_string is not None:
            root = ET.fromstring(xml_string)
            self.tree = ET.ElementTree(root)
        self.geometry = geo.Geo(self.tree)
        # If loading from file, set the geometry source path
        if (
            self.input_file is not None
            and self.geometry.has_geometry
            and self.geometry.filename
        ):
            gml_path = Path(self.input_file).parent / self.geometry.filename
            assert gml_path.exists()
            self.geometry._active_target = gml_path

        self.mesh = mesh.Mesh(self.tree)
        self.processes = processes.Processes(self.tree)
        self.python_script = python_script.PythonScript(self.tree)
        # If loading from file, set the python_script source path
        if self.input_file is not None and self.python_script.filename:
            py_path = Path(self.input_file).parent / self.python_script.filename
            if py_path.exists():
                self.python_script._active_target = py_path
        self.media = media.Media(self.tree)
        self.time_loop = timeloop.TimeLoop(self.tree)
        self.local_coordinate_system = (
            local_coordinate_system.LocalCoordinateSystem(self.tree)
        )
        self.parameters = parameters.Parameters(self.tree)
        self.curves = curves.Curves(self.tree)
        self.process_variables = processvars.ProcessVars(self.tree)
        self.nonlinear_solvers = nonlinsolvers.NonLinSolvers(self.tree)
        self.linear_solvers = linsolvers.LinSolvers(self.tree)

    @classmethod
    def from_folder(cls, folder: Path | str) -> Self:
        """
        Load Project from a folder containing a project.prj file.

        :param folder: Path to the project folder.
        :returns:      A Project instance loaded from the folder.
        """
        folder = Path(folder)
        prj_file = folder / "project.prj"
        if not prj_file.exists():
            msg = f"No project.prj found in {folder}"
            raise FileNotFoundError(msg)
        project = cls(input_file=prj_file)
        project._bind_to_path(folder)
        return project

    @classmethod
    def from_id(cls, project_id: str) -> Self:
        """
        Load Project from the user storage path using its ID.
        StorageBase.Userpath must be set.

        :param project_id: The unique ID of the Project to load.
        :returns:          A Project instance restored from disk.
        """
        project_folder = StorageBase.saving_path() / "Project" / project_id
        project_file = project_folder / "project.prj"

        if not project_file.exists():
            msg = f"No project found at {project_file}"
            raise FileNotFoundError(msg)

        project = cls(input_file=project_file)
        project._bind_to_path(project_folder)
        project._id = project_id
        return project

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        base_repr = super().__repr__()

        if self.user_specified_id:
            construct = f'{cls_name}.from_id("{self._id}")'
        elif self.is_saved:
            construct = f"{cls_name}.from_folder({str(self.active_target)!r})"
        else:
            # Show comprehensive view of object's data
            attrs = []
            if self.input_file:
                attrs.append(f"input_file={str(self.input_file)!r}")
            if self.output_dir != Path():
                attrs.append(f"output_dir={str(self.output_dir)!r}")
            if self.logfile != Path("out.log"):
                attrs.append(f"logfile={str(self.logfile)!r}")
            if self.verbose:
                attrs.append(f"verbose={self.verbose!r}")

            attrs_str = ", ".join(attrs) if attrs else ""
            construct = f"{cls_name}({attrs_str})"

        return f"{construct}\n{base_repr}"

    def __str__(self) -> str:
        base_repr = super().__str__()
        if self.input_file:
            dependencies = ",".join(
                str(self.dependencies(include_meshes=False))
            )
        else:
            dependencies = None
        lines = [
            f"{base_repr}",
            f"   Input file: {self.input_file!r}\n"
            f"   Dependencies: {dependencies!r}",
        ]

        return "\n".join(lines)

    def __deepcopy__(self, memo: dict) -> Self:
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new

        skip_attrs = self._SAVE_STATE_ATTRS + (
            "prjfile",
            "process",
            "geometry",
            "python_script",
        )
        for k, v in self.__dict__.items():
            if k not in skip_attrs:
                setattr(new, k, copy.deepcopy(v, memo))

        new._reset_save_state()
        new.prjfile = new._next_target / "project.prj"
        assert isinstance(new.prjfile, Path)

        # Create new Geo with the copied tree and preserve source_path
        new.geometry = copy.deepcopy(self.geometry)
        if self.geometry.filename:
            new.geometry._next_target = (
                new._next_target / self.geometry.filename
            )

        # Create new PythonScript with the copied tree and preserve source_path
        new.python_script = copy.deepcopy(self.python_script)
        if self.python_script.filename:
            new.python_script._next_target = (
                new._next_target / self.python_script.filename
            )
        return new

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Project):
            return NotImplemented

        self._remove_empty_elements()
        other._remove_empty_elements()

        c14n_a = ET.tostring(self.tree, method="c14n")
        c14n_b = ET.tostring(other.tree, method="c14n")
        if c14n_a == c14n_b:
            return True

        text_a = c14n_a.decode("utf-8")
        text_b = c14n_b.decode("utf-8")

        # Split into lines to get a readable diff
        lines_a = text_a.splitlines(keepends=True)
        lines_b = text_b.splitlines(keepends=True)

        diff = difflib.unified_diff(
            lines_a,
            lines_b,
            fromfile=str(self.active_target),
            tofile=str(self.active_target),
        )

        print("Files differ (canonical XML):")
        for line in diff:
            # difflib already prefixes lines with -/+/@
            sys.stdout.write(line)
        return False

    def _replace_blocks_by_includes(self) -> None:
        for i, file in enumerate(self.include_files):
            parent_element = self.include_elements[i].getparent()
            include_element = ET.SubElement(parent_element, "include")
            file_ = file if self.folder.cwd() else file.relative_to(self.folder)
            include_element.set("file", str(file_))
            parse = ET.XMLParser(remove_blank_text=True)
            include_string = ET.tostring(
                self.include_elements[i], pretty_print=True
            )
            include_parse = ET.fromstring(include_string, parser=parse)
            include_tree = ET.ElementTree(include_parse)
            ET.indent(include_tree, space="    ")
            include_tree.write(
                file,
                encoding="ISO-8859-1",
                xml_declaration=False,
                pretty_print=True,
            )
            parent_element.remove(self.include_elements[i])

    def _get_root(
        self, remove_blank_text: bool = False, remove_comments: bool = False
    ) -> ET._Element:
        parser = ET.XMLParser(
            remove_blank_text=remove_blank_text,
            remove_comments=remove_comments,
            huge_tree=True,
        )
        if self.tree is None:
            if self.input_file is not None:
                self.tree = ET.parse(str(self.input_file), parser)
            else:
                msg = "No inputfile given. Can't build XML tree."
                raise RuntimeError(msg)
        root = self.tree.getroot()
        all_occurrences = root.findall(".//include")
        for occurrence in all_occurrences:
            self.include_files.append(occurrence.attrib["file"])
        for i, _ in enumerate(all_occurrences):
            _tree = ET.parse(str(self.folder / self.include_files[i]), parser)
            _root = _tree.getroot()
            parentelement = all_occurrences[i].getparent()
            children_before = parentelement.getchildren()
            parentelement.append(_root)
            parentelement.remove(all_occurrences[i])
            children_after = parentelement.getchildren()
            for child in children_after:
                if child not in children_before:
                    self.include_elements.append(child)
        return root

    def _remove_empty_elements(self) -> None:
        root = self._get_root()
        empty_text_list = ["./geometry", "./python_script"]
        empty_el_list = [
            "./time_loop/global_process_coupling",
            "./curves",
            "./media",
            "./local_coordinate_system",
        ]
        for element in empty_text_list:
            entry = root.find(element)
            if entry is not None:
                self.remove_element(".", tag=entry.tag, text="")
            if entry is not None:
                self.remove_element(".", tag=entry.tag, text=None)
        for element in empty_el_list:
            entry = root.find(element)
            if (entry is not None) and (len(entry.getchildren()) == 0):
                entry.getparent().remove(entry)

    def dependencies(
        self,
        include_meshes: bool = True,
        include_python_file: bool = True,
        include_xml_includes: bool = True,
    ) -> list[Path]:

        return Project._dependencies(
            self.tree, include_meshes, include_python_file, include_xml_includes
        )

    @staticmethod
    def _dependencies(
        root: ET._Element,
        include_meshes: bool = True,
        include_python_file: bool = True,
        include_xml_includes: bool = True,
    ) -> list[Path]:
        # root = tree.getroot()
        if include_xml_includes:
            xml_files = [
                Path(elem.get("file"))
                for elem in root.findall(".//include")
                if elem.get("file")
            ]
        else:
            xml_files = []

        if include_meshes:
            mesh_files = [
                Path(m.text)
                for xpath in ["./mesh", "./meshes/mesh", "./geometry"]
                for m in root.findall(xpath)
            ]
        else:
            mesh_files = []

        if include_python_file:
            python_files = [
                Path(m.text) for m in root.findall("./python_script") if m.text
            ]
        else:
            python_files = []

        # combine, make unique, preserve order
        return list(dict.fromkeys(xml_files + mesh_files + python_files))

    @staticmethod
    def dependencies_of_file(
        input_file: str | Path,
        mesh_dir: str | Path | None = None,
        check: bool = False,
        include_meshes: bool = True,
        include_python_file: bool = True,
        include_xml_includes: bool = True,
    ) -> list[Path]:
        """
        Searches a (partial) project file for included files (e.g. xml snippets, meshes (vtu,gml), python scripts)
        Can be used before constructing a Project object (static function).

        :param input_file:  Path to the prj-file
        :param mesh_dir:    Optional directory used to resolve referenced mesh files.
                            If omitted, mesh paths is interpreted to be in same folder as input_file.
        :param check:       If `True`, assert that all collected files exist on disk
        :returns:           A list of dependency file paths (order-preserving, de-duplicated).
        :raises AssertionError:
                            If `check=True` and at least one referenced file is missing.

        """

        input_file = Path(input_file)
        if not mesh_dir:
            mesh_dir = input_file.parent
        mesh_dir = Path(mesh_dir).absolute()

        tree = ET.parse(input_file)
        files = Project._dependencies(
            tree,
            include_meshes,
            include_python_file,
            include_xml_includes,
        )
        if check:
            missing_files = [file for file in files if not file.exists()]
            if len(missing_files) > 0:
                missing_files_str = ", ".join(str(missing_files))
                msg = f"The following files are missing: {missing_files_str}."
                raise FileExistsError(msg)
        return files

    @staticmethod
    def _get_parameter_pointer(
        root: ET._Element, name: str, xpath: str
    ) -> ET._Element:
        params = root.findall(xpath)
        parameterpointer = None
        for parameter in params:
            for paramproperty in parameter:
                if (paramproperty.tag == "name") and (
                    paramproperty.text == name
                ):
                    parameterpointer = parameter
        if parameterpointer is None:
            msg = "Parameter/Property not found"
            raise RuntimeError(msg)
        return parameterpointer

    @staticmethod
    def _get_medium_pointer(root: ET._Element, mediumid: int) -> ET._Element:
        xpathmedia = "./media/medium"
        mediae = root.findall(xpathmedia)
        mediumpointer = None
        for medium in mediae:
            try:
                if medium.attrib["id"] == str(mediumid):
                    mediumpointer = medium
            except KeyError:
                if (len(mediae) == 1) and (str(mediumid) == "0"):
                    mediumpointer = medium
        if mediumpointer is None:
            msg = "Medium not found"
            raise RuntimeError(msg)
        return mediumpointer

    @staticmethod
    def _get_phase_pointer(root: ET._Element, phase: str) -> ET._Element:
        phases = root.findall("./phases/phase")
        phasetypes = root.findall("./phases/phase/type")
        phasecounter = None
        for i, phasetype in enumerate(phasetypes):
            if phasetype.text == phase:
                phasecounter = i
        phasepointer = phases[phasecounter]
        if phasepointer is None:
            msg = "Phase not found"
            raise RuntimeError(msg)
        return phasepointer

    @staticmethod
    def _get_component_pointer(
        root: ET._Element, component: str
    ) -> ET._Element:
        components = root.findall("./components/component")
        componentnames = root.findall("./components/component/name")
        componentcounter = None
        for i, componentname in enumerate(componentnames):
            if componentname.text == component:
                componentcounter = i
        componentpointer = components[componentcounter]
        if componentpointer is None:
            msg = "Component not found"
            raise RuntimeError(msg)
        return componentpointer

    @staticmethod
    def _set_type_value(
        parameterpointer: ET._ElementIterator,
        value: int,
        parametertype: Any | None,
        valuetag: str | None = None,
    ) -> None:
        for paramproperty in parameterpointer:
            if (paramproperty.tag == valuetag) and (value is not None):
                paramproperty.text = str(value)
            elif paramproperty.tag == "type" and parametertype is not None:
                paramproperty.text = str(parametertype)

    def add_element(
        self,
        parent_xpath: str = "./",
        tag: str | None = None,
        text: str | int | float | None = None,
        attrib_list: list[str] | None = None,
        attrib_value_list: list[str] | None = None,
    ) -> None:
        """General method to add an Entry.

        An element is a single tag containing 'text',
        attributes and anttribute values.

        :param parent_xpath:        XPath of the parent tag
        :param tag:                 tag name
        :param text:                content
        :param attrib_list:         list of attribute keywords
        :param attrib_value_list:   list of values of the attribute keywords
        """
        root = self._get_root()
        parents = root.findall(parent_xpath)
        for parent in parents:
            if tag is not None:
                q = ET.SubElement(parent, tag)
                if text is not None:
                    q.text = str(text)
                if attrib_list is not None:
                    if attrib_value_list is None:
                        msg = "attrib_value_list is not given for add_element"
                        raise RuntimeError(msg)
                    if len(attrib_list) != len(attrib_value_list):
                        msg = "The size of attrib_list is not the same as that of attrib_value_list"
                        raise RuntimeError(msg)

                    for attrib, attrib_value in zip(
                        attrib_list, attrib_value_list, strict=False
                    ):
                        q.set(attrib, attrib_value)

    def add_include(self, parent_xpath: str = "./", file: str = "") -> None:
        """Add include element.

        :param parent_xpath: XPath of the parent tag
        :param file:         file name
        """
        self.add_includes.append({"parent_xpath": parent_xpath, "file": file})

    def _add_includes(self, root: ET._Element) -> None:
        for add_include in self.add_includes:
            parent = root.findall(add_include["parent_xpath"])
            newelement = []
            for i, entry in enumerate(parent):
                newelement.append(ET.SubElement(entry, "include"))
                newelement[i].set("file", add_include["file"])

    def add_block(
        self,
        blocktag: str,
        block_attrib: Any | None = None,
        parent_xpath: str = "./",
        taglist: list[str] | None = None,
        textlist: list[Any] | None = None,
    ) -> None:
        """General method to add a Block.

        A block consists of an enclosing tag containing a number of
        subtags retaining a key-value structure.

        :param blocktag:     name of the enclosing tag
        :param block_attrib: attributes belonging to the blocktag
        :param parent_xpath: XPath of the parent tag
        :param taglist:      list of strings containing the keys
        :param textlist:     list retaining the corresponding values

        """
        root = self._get_root()
        parents = root.findall(parent_xpath)
        for parent in parents:
            q = ET.SubElement(parent, blocktag)
            if block_attrib is not None:
                for key, val in block_attrib.items():
                    q.set(key, val)
            if (taglist is not None) and (textlist is not None):
                for i, tag in enumerate(taglist):
                    r = ET.SubElement(q, tag)
                    if textlist[i] is not None:
                        r.text = str(textlist[i])

    def deactivate_property(
        self, name: str, mediumid: int = 0, phase: str | None = None
    ) -> None:
        """Replaces MPL properties by a comment.

        :param mediumid: id of the medium
        :param phase:    name of the phase
        :param name:     property name
        """
        root = self._get_root()
        mediumpointer = self._get_medium_pointer(root, mediumid)
        xpathparameter = "./properties/property"
        if phase is None:
            parameterpointer = self._get_parameter_pointer(
                mediumpointer, name, xpathparameter
            )
        else:
            phasepointer = self._get_phase_pointer(mediumpointer, phase)
            parameterpointer = self._get_parameter_pointer(
                phasepointer, name, xpathparameter
            )
        parameterpointer.getparent().replace(
            parameterpointer, ET.Comment(ET.tostring(parameterpointer))
        )

    def deactivate_parameter(self, name: str) -> None:
        """Replaces parameters by a comment.

        :param name: property name
        """
        root = self._get_root()
        parameterpath = "./parameters/parameter"
        parameterpointer = self._get_parameter_pointer(
            root, name, parameterpath
        )
        parameterpointer.getparent().replace(
            parameterpointer, ET.Comment(ET.tostring(parameterpointer))
        )

    def remove_element(
        self, xpath: str, tag: str | None = None, text: str | None = None
    ) -> None:
        """Removes an element.

        :param xpath:
        :param tag:
        :param text:
        """
        root = self._get_root()
        elements = root.findall(xpath)
        if tag is None:
            for element in elements:
                element.getparent().remove(element)
        else:
            for element in elements:
                sub_elements = element.getchildren()
                for sub_element in sub_elements:
                    if sub_element.tag == tag and sub_element.text == text:
                        sub_element.getparent().remove(sub_element)

    def replace_text(
        self, value: str | int, xpath: str = ".", occurrence: int = -1
    ) -> None:
        """General method for replacing text between opening and closing tags.

        :param value:      Text
        :param xpath:      XPath of the tag
        :param occurrence: Easy way to address nonunique XPath addresses by
                           their occurrence from the top of the XML file
        """
        root = self._get_root()
        find_xpath = root.findall(xpath)
        for i, entry in enumerate(find_xpath):
            if occurrence < 0 or i == occurrence:
                entry.text = str(value)

    def replace_block_by_include(
        self,
        xpath: str = "./",
        filename: str = "include.xml",
        occurrence: int = 0,
    ) -> None:
        """General method for replacing a block by an include.

        :param xpath:      XPath of the tag
        :param filename:   name of the include file
        :param occurrence: Addresses nonunique XPath by their occurece
        """
        print(
            "Note: Includes are only written if write_input(keep_includes=True) is called."
        )
        root = self._get_root()
        find_xpath = root.findall(xpath)
        for i, entry in enumerate(find_xpath):
            if i == occurrence:
                self.include_elements.append(entry)
                self.include_files.append(self.prjfile.parent / filename)

    def replace_mesh(self, oldmesh: str, newmesh: str) -> None:
        """Method to replace meshes.

        :param oldmesh:
        :param newmesh:
        """
        root = self._get_root()
        bulkmesh = root.find("./mesh")
        if bulkmesh is not None:
            if bulkmesh.text == oldmesh:
                bulkmesh.text = newmesh
            else:
                msg = "Bulk mesh name and oldmesh argument don't agree."
                raise RuntimeError(msg)
        all_occurrences_meshsection = root.findall("./meshes/mesh")
        for occurrence in all_occurrences_meshsection:
            if occurrence.text == oldmesh:
                occurrence.text = newmesh
        all_occurrences = root.findall(".//mesh")
        for occurrence in all_occurrences:
            if occurrence not in all_occurrences_meshsection:
                oldmesh_stripped = os.path.split(oldmesh)[1].replace(".vtu", "")
                newmesh_stripped = os.path.split(newmesh)[1].replace(".vtu", "")
                if occurrence.text == oldmesh_stripped:
                    occurrence.text = newmesh_stripped

    def replace_parameter(
        self,
        name: str = "",
        parametertype: str = "",
        taglist: list[str] | None = None,
        textlist: list[str] | None = None,
    ) -> None:
        """Replacing parametertypes and values.

        :param name:          parametername
        :param parametertype: parametertype
        :param taglist:       list of tags needed for parameter spec
        :param textlist:      values of parameter
        """
        root = self._get_root()
        parameterpath = "./parameters/parameter[name='" + name + "']"
        parent = root.find(parameterpath)
        children = parent.getchildren()
        for child in children:
            if child.tag not in ["name", "type"]:
                self.remove_element(f"{parameterpath}/{child.tag}")
        paramtype = root.find(f"{parameterpath}/type")
        paramtype.text = parametertype
        if (taglist is not None) and (textlist is not None):
            for i, tag in enumerate(taglist):
                if tag not in ["name", "type"]:
                    self.add_element(
                        parent_xpath=parameterpath, tag=tag, text=textlist[i]
                    )

    def replace_parameter_value(
        self, name: str = "", value: int = 0, valuetag: str = "value"
    ) -> None:
        """Replacing parameter values.

        :param name:          parametername
        :param value:         value
        :param parametertype: parameter type
        :param valuetag:      name of the tag containing the value, e.g., values
        """
        root = self._get_root()
        parameterpath = "./parameters/parameter"
        parameterpointer = self._get_parameter_pointer(
            root, name, parameterpath
        )
        self._set_type_value(parameterpointer, value, None, valuetag=valuetag)

    def replace_phase_property_value(
        self,
        mediumid: int = 0,
        phase: str = "AqueousLiquid",
        component: str | None = None,
        name: str = "",
        value: int = 0,
        propertytype: str = "Constant",
        valuetag: str = "value",
    ) -> None:
        """Replaces properties in medium phases.

        :param mediumid:     id of the medium
        :param phase:        name of the phase
        :param component:    name of the component
        :param name:         property name
        :param value:        value
        :param propertytype: type of the property
        :param valuetag:     name of the tag containing the value, e.g., values
        """
        root = self._get_root()
        mediumpointer = self._get_medium_pointer(root, mediumid)
        phasepointer = self._get_phase_pointer(mediumpointer, phase)
        if component is not None:
            phasepointer = self._get_component_pointer(phasepointer, component)
        xpathparameter = "./properties/property"
        parameterpointer = self._get_parameter_pointer(
            phasepointer, name, xpathparameter
        )
        self._set_type_value(
            parameterpointer, value, propertytype, valuetag=valuetag
        )

    def replace_medium_property_value(
        self,
        mediumid: int = 0,
        name: str = "",
        value: int = 0,
        propertytype: str = "Constant",
        valuetag: str = "value",
    ) -> None:
        """Replaces properties in medium (not belonging to any phase).

        :param mediumid:     id of the medium
        :param name:         property name
        :param value:        value
        :param propertytype: type of the property
        :param valuetag:     name of the tag containing the value, e.g., values
        """
        root = self._get_root()
        mediumpointer = self._get_medium_pointer(root, mediumid)
        xpathparameter = "./properties/property"
        parameterpointer = self._get_parameter_pointer(
            mediumpointer, name, xpathparameter
        )
        self._set_type_value(
            parameterpointer, value, propertytype, valuetag=valuetag
        )

    def set(self, **args: str | int) -> None:
        """
        Sets directly a uniquely defined property.
        List of properties is given in the dictory below.
        """
        property_db = {
            "t_initial": "./time_loop/processes/process/time_stepping/t_initial",
            "t_end": "./time_loop/processes/process/time_stepping/t_end",
            "output_prefix": "./time_loop/output/prefix",
            "reltols": "./time_loop/processes/process/convergence_criterion/reltols",
            "abstols": "./time_loop/processes/process/convergence_criterion/abstols",
            "mass_lumping": "./processes/process/mass_lumping",
            "eigen_solver": "./linear_solvers/linear_solver/eigen/solver_type",
            "eigen_precon": "./linear_solvers/linear_solver/eigen/precon_type",
            "eigen_max_iteration_step": "./linear_solvers/linear_solver/eigen/max_iteration_step",
            "eigen_error_tolerance": "./linear_solvers/linear_solver/eigen/error_tolerance",
            "eigen_scaling": "./linear_solvers/linear_solver/eigen/scaling",
            "petsc_prefix": "./linear_solvers/linear_solver/petsc/prefix",
            "petsc_parameters": "./linear_solvers/linear_solver/petsc/parameters",
            "compensate_displacement": "./process_variables/process_variable[name='displacement']/compensate_non_equilibrium_initial_residuum",
            "compensate_all": "./process_variables/process_variable/compensate_non_equilibrium_initial_residuum",
        }
        for key, val in args.items():
            self.replace_text(val, xpath=property_db[key])

    def restart(
        self,
        restart_suffix: str = "_restart",
        t_initial: float | None = None,
        t_end: float | None = None,
        zero_displacement: bool = False,
    ) -> None:
        """Prepares the project file for a restart.

        Takes the last time step from the PVD file mentioned in the PRJ file.
        Sets initial conditions accordingly.

        :param restart_suffix:    suffix by which the output prefix is appended
        :param t_initial:         first time step, takes the last from previous simulation if None
        :param t_end:             last time step, the same as in previous run if None
        :param zero_displacement: sets the initial displacement to zero if True
        """

        root_prj = self._get_root()
        filetype = root_prj.find("./time_loop/output/type").text
        pvdfile = root_prj.find("./time_loop/output/prefix").text + ".pvd"
        pvdfile = self.output_dir / pvdfile
        if filetype != "VTK":
            msg = "Output file type unknown. Please use VTK."
            raise RuntimeError(msg)
        tree = ET.parse(pvdfile)
        xpath = "./Collection/DataSet"
        root_pvd = tree.getroot()
        find_xpath = root_pvd.findall(xpath)
        lastfile = find_xpath[-1].attrib["file"]
        last_time = find_xpath[-1].attrib["timestep"]
        try:
            bulk_mesh = root_prj.find("./mesh").text
        except AttributeError:
            try:
                bulk_mesh = root_prj.find("./meshes/mesh").text
            except AttributeError:
                print("Can't find bulk mesh.")
        self.replace_mesh(bulk_mesh, lastfile)
        root_prj.find("./time_loop/output/prefix").text = (
            root_prj.find("./time_loop/output/prefix").text + restart_suffix
        )
        t_initials = root_prj.findall(
            "./time_loop/processes/process/time_stepping/t_initial"
        )
        t_ends = root_prj.findall(
            "./time_loop/processes/process/time_stepping/t_end"
        )
        for i, t0 in enumerate(t_initials):
            if t_initial is None:
                t0.text = last_time
            else:
                t0.text = str(t_initial)
            if t_end is not None:
                t_ends[i].text = str(t_end)
        process_vars = root_prj.findall(
            "./process_variables/process_variable/name"
        )
        ic_names = root_prj.findall(
            "./process_variables/process_variable/initial_condition"
        )
        for i, process_var in enumerate(process_vars):
            if process_var.text == "displacement" and zero_displacement is True:
                print(
                    "Please make sure that epsilon_ip is removed from the VTU file before you run OGS."
                )
                zero = {"1": "0", "2": "0 0", "3": "0 0 0"}
                cpnts = root_prj.find(
                    "./process_variables/process_variable[name='displacement']/components"
                ).text
                self.replace_parameter(
                    name=ic_names[i].text,
                    parametertype="Constant",
                    taglist=["values"],
                    textlist=[zero[cpnts]],
                )
            else:
                self.replace_parameter(
                    name=ic_names[i].text,
                    parametertype="MeshNode",
                    taglist=["mesh", "field_name"],
                    textlist=[
                        lastfile.split("/")[-1].replace(".vtu", ""),
                        process_var.text,
                    ],
                )
        self.remove_element("./processes/process/initial_stress")

    def _get_output_file(self) -> Path:
        """Helper method to extract output filename from the project."""
        if self.tree is None:
            msg = "self.tree is empty."
            raise AttributeError(msg)

        mesh = self.tree.findtext("./mesh") or self.tree.findtext(
            "./meshes/mesh"
        )
        if mesh is None:
            msg = "Expected <mesh> definition."
            raise AttributeError(msg)

        fn_type = self.tree.findtext("./time_loop/output/type").strip()
        prefix = self.tree.findtext("./time_loop/output/prefix", "")
        prefix = prefix.replace("{:meshname}", Path(mesh).stem)

        fn: Path
        match fn_type:
            case "VTK":
                fn = self.output_dir / f"{prefix}.pvd"
            case "XDMF":
                fn = self.output_dir / f"{prefix}_{Path(mesh).stem}.xdmf"
            case _:
                msg = "Output file type unknown. Please use VTK/XDMF."
                raise RuntimeError(msg)
        return fn

    def _write_prj_to_pvd(self) -> None:
        self.input_file = self.prjfile
        root = self._get_root(remove_blank_text=True, remove_comments=True)
        prjstring = (
            ET.tostring(root)
            .decode("utf-8", errors="ignore")
            .replace("\r", " ")
            .replace("\n", " ")
            .replace("\t", " ")
            .replace("--", "")
        )

        fn = self._get_output_file()
        if not fn.exists():
            msg = f"Specified output file not found: {fn}."
            raise FileNotFoundError(msg)

        tree_pvd = ET.parse(fn)
        root_pvd = tree_pvd.getroot()
        root_pvd.append(ET.Comment(prjstring))
        tree_pvd.write(
            fn,
            encoding="ISO-8859-1",
            xml_declaration=True,
            pretty_print=True,
        )
        print("Project file written to output.")
        return

    def _failed_run_print_log_tail(
        self, write_logs: bool, tail_len: int = 10
    ) -> str:
        msg = "OGS execution was not successful."
        if write_logs is False:
            msg += " Please set write_logs to True to obtain more information."
        else:
            print(f"Last {tail_len} line of the log:")
            with self.logfile.open() as lf:
                last_lines = "".join(lf.readlines()[-tail_len:])
                msg += last_lines

        return msg

    def run_model(
        self,
        logfile: Path | None = Path("out.log"),
        path: Path | None = None,
        args: Any | None = None,
        container_path: Path | str | None = None,
        wrapper: Any | None = None,
        write_logs: bool = True,
        background: bool = False,
    ) -> "subprocess.Popen":
        """Command to run OGS.

        Runs OGS with the project file specified as output_file.

        :param logfile: Name of the file to write STDOUT of ogs
        :param path:    Path of the directory in which the ogs executable can be found.
                       If ``container_path`` is given: Path to the directory in which the
                       Singularity executable can be found.
        :param args:   additional arguments for the ogs executable
        :param container_path:   Path of the OGS container file.
        :param wrapper:          add a wrapper command. E.g. mpirun
        :param write_logs:       set False to omit logging
        :param write_prj_to_pvd: write the prj file as a comment in the pvd
        :param background:       Run the simulation in a background process
        """

        ogs_path: Path = Path()
        env = os.environ.copy()
        if self.threads is not None:
            env["OMP_NUM_THREADS"] = f"{self.threads}"
        if self.asm_threads is not None:
            env["OGS_ASM_THREADS"] = f"{self.asm_threads}"
        if container_path is not None:
            container_path = Path(container_path)
            container_path = container_path.expanduser()
            if not container_path.is_file():
                msg = """The specific container-path is not a file. \
                        Please provide a path to the OGS container."""
                raise RuntimeError(msg)
            if str(container_path.suffix).lower() != ".sif":
                msg = """The specific file is not a Singularity container. \
                        Please provide a *.sif file containing OGS."""
                raise RuntimeError(msg)

        if path is None:
            ogs_path_env = os.getenv("OGS_BIN_PATH", None)
            if ogs_path_env is not None:
                path = Path(ogs_path_env)

        if path:
            path = Path(path)
            path = path.expanduser()
            if not path.is_dir():
                if container_path is not None:
                    msg = """The specified path is not a directory. \
                            Please provide a directory containing the Singularity executable."""
                    raise RuntimeError(msg)
                msg = """The specified path is not a directory. \
                        Please provide a directory containing the OGS executable."""
                raise RuntimeError(msg)
            ogs_path = ogs_path / path
        # TODO: logfile should also be written in self.output_path by default
        # or path provided via "-o"
        # Fix this, when argparse is used to read the args
        if logfile is not None:
            self.logfile = Path(logfile)
        if container_path is not None:
            if sys.platform == "win32":
                msg = """Running OGS in a Singularity container is only possible in Linux.\
                        See https://sylabs.io/guides/3.0/user-guide/installation.html\
                        for Windows solutions."""
                raise RuntimeError(msg)
            ogs_path = ogs_path / "singularity"
            if shutil.which(str(ogs_path)) is None:
                msg = """The Singularity executable was not found.\
                        See https://www.opengeosys.org/docs/userguide/basics/container/\
                        for installation instructions."""
                raise RuntimeError(msg)
        else:
            if sys.platform == "win32":
                ogs_path = ogs_path / "ogs.exe"
            else:
                ogs_path = ogs_path / "ogs"
            if shutil.which(str(ogs_path)) is None:
                msg = """The OGS executable was not found.\
                        See https://www.opengeosys.org/docs/userguide/basics/introduction/\
                        for installation instructions."""
                raise RuntimeError(msg)
        cmd = " "
        if wrapper is not None:
            cmd += wrapper + " "
        cmd += f"{ogs_path} "
        if container_path is not None:
            if wrapper is not None:
                cmd = "singularity exec " + f"{container_path} " + wrapper + " "
            else:
                cmd = "singularity exec " + f"{container_path} " + "ogs "
        # TODO: use argparse here
        if args is None:
            args = f" -o {self.output_dir}"
        elif args is not None and "-o" not in args:
            args += f" -o {self.output_dir}"

        if args is not None:
            argslist = args.split(" ")
            if "-o" in argslist:
                index = argslist.index("-o")
                if index + 1 < len(argslist):
                    self.output_dir = Path(argslist[index + 1])

            cmd += f"{args} "
        #        if write_logs is True:
        #            cmd += f"{self.prjfile} > {self.logfile}"
        #        else:
        cmd += f"{self.prjfile}"
        if sys.platform == "win32":
            executable = "C:\\Windows\\System32\\cmd.exe"
        else:
            executable = "bash"

        self.runtime_start = time.time()
        if write_logs is True:
            with self.logfile.open("w") as logf:
                self.process = subprocess.Popen(
                    cmd,
                    shell=True,
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
        else:
            self.process = subprocess.Popen(
                cmd,
                shell=True,
                executable=executable,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                env=env,
            )
        if not background:
            self.process.wait()

        return self.process

    def _propagate_target(self) -> None:
        """Propagate save target to geometry and python_script files."""
        if self.geometry.filename and not self.geometry.user_specified_target:
            self.geometry._next_target = (
                self.next_target / self.geometry.filename
            )
        if (
            self.python_script.filename
            and not self.python_script.user_specified_target
        ):
            self.python_script._next_target = (
                self.next_target / self.python_script.filename
            )

    def write_input(
        self,
        prjfile_path: None | Path | str = None,
        keep_includes: bool = False,
    ) -> None:
        """Writes the projectfile to disk.

        :param prjfile_path: Path to write the project file to. If not specified, the initialised path is used.
        :param keep_includes:
        """
        prjfile_path = Path(prjfile_path) if prjfile_path else self.prjfile
        prjfile_path.parent.mkdir(parents=True, exist_ok=True)

        if self.tree is not None:
            self._remove_empty_elements()
            if keep_includes is True:
                self._replace_blocks_by_includes()
            root = self.tree.getroot()
            self._add_includes(root)
            parse = ET.XMLParser(remove_blank_text=True)
            self.tree_string = ET.tostring(root, pretty_print=True)
            self.tree = ET.ElementTree(
                ET.fromstring(self.tree_string, parser=parse)
            )
            ET.indent(self.tree, space="    ")
            if self.verbose is True:
                display.Display(self.tree)
            self.tree.write(
                prjfile_path,
                encoding="ISO-8859-1",
                xml_declaration=True,
                pretty_print=True,
            )
        else:
            msg = "No tree has been build."
            raise RuntimeError(msg)

    def _property_df_move_elastic_properties_to_mpl(
        self, newtree: ET._ElementTree, root: ET._Element
    ) -> None:
        for entry in newtree.findall(
            "./processes/process/constitutive_relation"
        ):
            medium = self._get_medium_pointer(root, entry.attrib.get("id", "0"))
            parent = medium.find("./phases/phase[type='Solid']/properties")
            taglist = ["name", "type", "parameter_name"]
            for subentry in entry:
                if subentry.tag in [
                    "youngs_modulus",
                    "poissons_ratio",
                    "youngs_moduli",
                    "poissons_ratios",
                    "shear_moduli",
                ]:
                    textlist = [subentry.tag, "Parameter", subentry.text]
                    q = ET.SubElement(parent, "property")
                    for i, tag in enumerate(taglist):
                        r = ET.SubElement(q, tag)
                        if textlist[i] is not None:
                            r.text = str(textlist[i])

    def _resolve_parameters(
        self, location: str, newtree: ET._ElementTree
    ) -> None:
        parameter_names_add = newtree.findall(
            f"./media/medium/{location_pointer[location]}properties/property[type='Parameter']/parameter_name"
        )
        parameter_names = [name.text for name in parameter_names_add]
        for parameter_name in parameter_names:
            param_type = newtree.find(
                f"./parameters/parameter[name='{parameter_name}']/type"
            ).text
            if param_type == "Constant":
                param_value = newtree.findall(
                    f"./parameters/parameter[name='{parameter_name}']/value"
                )
                param_value.append(
                    newtree.find(
                        f"./parameters/parameter[name='{parameter_name}']/values"
                    )
                )
                property_type = newtree.findall(
                    f"./media/medium/{location_pointer[location]}properties/property[parameter_name='{parameter_name}']/type"
                )
                for entry in property_type:
                    entry.text = "Constant"
                property_value = newtree.findall(
                    f"./media/medium/{location_pointer[location]}properties/property[parameter_name='{parameter_name}']/parameter_name"
                )
                for entry in property_value:
                    entry.tag = "value"
                    entry.text = param_value[0].text

    def _generate_property_list(
        self,
        mediamapping: dict[int, str],
        numofmedia: int,
        root: ET._Element,
        property_names: list,
        values: dict[str, list],
        location: str,
        property_list: list,
    ) -> list:
        for name in property_names:
            values[name] = []
            orig_name = "".join(c for c in name if not c.isnumeric())
            number_suffix = "".join(c for c in name if c.isnumeric())
            if orig_name in property_dict[location]:
                for medium_id in range(numofmedia):
                    if medium_id in mediamapping:
                        medium = self._get_medium_pointer(root, medium_id)
                        proptytype = medium.find(
                            f"./{location_pointer[location]}properties/property[name='{name}']/type"
                        )
                        if proptytype is None:
                            values[name].append(
                                Value(mediamapping[medium_id], None)
                            )
                        else:
                            if proptytype.text == "Constant":
                                value_entry = medium.find(
                                    f"./{location_pointer[location]}properties/property[name='{name}']/value"
                                ).text
                                value_entry_list = value_entry.split(" ")
                                if len(value_entry_list) == 1:
                                    values[name].append(
                                        Value(
                                            mediamapping[medium_id],
                                            float(value_entry),
                                        )
                                    )
                            else:
                                values[name].append(
                                    Value(mediamapping[medium_id], None)
                                )
                if number_suffix != "":
                    new_symbol = (
                        property_dict[location][orig_name]["symbol"][:-1]
                        + "_"
                        + number_suffix
                        + "$"
                    )
                else:
                    new_symbol = property_dict[location][orig_name]["symbol"]
                property_list.append(
                    Property(
                        property_dict[location][orig_name]["title"],
                        new_symbol,
                        property_dict[location][orig_name]["unit"],
                        values[name],
                    )
                )
        return property_list

    def property_dataframe(
        self, mediamapping: dict[int, str] | None = None
    ) -> pd.DataFrame:
        """Returns a dataframe containing most properties
        defined in the Material Property (MPL) section of
        the input file.

        :param mediamapping:
        """
        newtree = copy.deepcopy(self.tree)
        if (newtree is None) or (self.tree is None):
            msg = "No tree existing."
            raise AttributeError(msg)
        root = newtree.getroot()
        property_list: list[Property] = []
        multidim_prop: dict[int, dict] = {}
        numofmedia = len(self.tree.findall("./media/medium"))
        if mediamapping is None:
            mediamapping = {}
            for i in range(numofmedia):
                mediamapping[i] = f"medium {i}"
        for i in range(numofmedia):
            multidim_prop[i] = {}
        self._property_df_move_elastic_properties_to_mpl(newtree, root)
        for location in location_pointer:
            self._resolve_parameters(location, newtree)
            _expand_tensors(self, numofmedia, multidim_prop, root, location)
            _expand_van_genuchten(self, numofmedia, root, location)
            property_names = [
                name.text
                for name in newtree.findall(
                    f"./media/medium/{location_pointer[location]}properties/property/name"
                )
            ]
            property_names = list(dict.fromkeys(property_names))
            values: dict[str, list] = {}
            property_list = self._generate_property_list(
                mediamapping,
                numofmedia,
                root,
                property_names,
                values,
                location,
                property_list,
            )
        properties = PropertySet(property=property_list)
        return pd.DataFrame(properties)

    def write_property_latextable(
        self,
        latexfile: Path = Path("property_dataframe.tex"),
        mediamapping: dict[int, str] | None = None,
        float_format: str = "{:.2e}",
    ) -> None:
        """Write material properties to disc
        as latex table.

        :param latexfile:
        :param mediamapping:
        :param float_format:
        """
        with latexfile.open("w") as tf:
            tf.write(
                self.property_dataframe(mediamapping).to_latex(
                    index=False, float_format=float_format.format
                )
            )

    def set_media(self, media_set: MediaSet) -> None:
        """Public API: import MediaSet into this Project."""
        _ProjectMediaImporter(self).set_media(media_set)

    def meshpaths(self, mesh_dir: Path | None = None) -> list[Path]:
        """Returns the filepaths to the given meshes in the Project.

        This does not include meshes defined via a .gml file.

        :param mesh_dir:    Path to the meshes directory (default: input dir)
        """
        mesh_dir = self.folder if mesh_dir is None else mesh_dir
        return [
            mesh_dir / m.text
            for xpath in ["./mesh", "./meshes/mesh"]
            for m in self._get_root().findall(xpath)
        ]

    def param_value_expression(self, param_name: str) -> str:
        """Return the text of the parameter value/s or expression.

        :param param_name:  Name of the parameter whose value is returned.
        """
        param = self._get_root().find(
            f"./parameters/parameter[name='{param_name}']"
        )
        if param is None:
            return param_name
        param_def = {e.tag: e.text for e in list(param.iter())[1:]}
        for tag in ["value", "values", "expression"]:
            if tag in param_def:
                return param_def[tag].strip().strip("\n")
        return param_name

    def constraints(self) -> dict[str, dict[str, list]]:
        """Creates a dict of boundary conditions and source terms.

        Structured in the following way:
        {meshname: {process_variable_name: [constraint_data]}}
        """
        root = self._get_root(remove_blank_text=True, remove_comments=True)

        def texts(xpath: str) -> list[str]:
            return [e.text for e in root.findall(xpath)]

        used_meshes = set(
            texts(".//boundary_condition/mesh")
            + texts(".//boundary_condition/geometry")
            + texts(".//source_term/mesh")
        )

        result: dict[str, dict[str, list]] = {
            mesh: {pvar: [] for pvar in texts(".//process_variable/name")}
            for mesh in used_meshes
        }

        for pvar in texts(".//process_variable/name"):
            for constraint_type in ["boundary_condition", "source_term"]:
                for entry in root.findall(
                    f".//process_variable[name='{pvar}']//{constraint_type}"
                ):
                    constraint = {
                        e.tag: e.text
                        for e in list(entry.iter())[1:]
                        if "comment" not in str(e.tag).lower()
                    }
                    if "mesh" in constraint:
                        mesh = constraint.pop("mesh")
                    else:
                        mesh = constraint.pop("geometry")
                        constraint.pop("geometrical_set")
                    if constraint_type == "source_term":
                        constraint["type"] += " source term"
                    result[mesh][pvar].append(constraint)

        return result

    def _format_constraint(self, constraint: dict, process_var: str) -> str:
        pvar_map = {
            "displacement": "$u$",
            "temperature": "$T$",
            "pressure": "$p$",
            "gas_pressure": "$p_g$",
            "capillary_pressure": "$p_c$",
        }
        comp_map = {"0": "$_x$", "1": "$_y$", "2": "$_z$"}
        non_comp_pvars = ["pressure", "temperature"]

        if (
            any(pvar in process_var for pvar in non_comp_pvars)
            and "component" in constraint
        ):
            constraint.pop("component")

        label = "\n  "
        constraint_type = constraint.get("type", "")
        if constraint_type == "Neumann" or "source term" in constraint_type:
            var_str = "~$d${pvar}{comp}/$dN$"
        else:
            var_str = "{pvar}{comp}"
        label += var_str.format(
            pvar=pvar_map.get(process_var, process_var),
            comp=comp_map.get(constraint.pop("component", ""), ""),
        )

        constraint_type = constraint.pop("type", "")
        if constraint_type not in ["", "Dirichlet", "Neumann"]:
            label += f" ({constraint_type})"
        if "parameter" in constraint:
            label += "=" + self.param_value_expression(
                constraint.pop("parameter")
            )
        for tag, text in constraint.items():
            text_val = self.param_value_expression(text)
            label += f"\n  {tag}: {text_val}"
        return label

    def constraints_labels(self) -> dict[str, str]:
        """Formatted information about boundary conditions and source terms.

        :returns:   Formatted str of constraints per meshname.

        Example output:

        .. code-block:: python

            {
                "domain": "domain:\\n  $T$=-8./3600*t+277.15\\n  $p$=0",
                "bottom": "bottom:\\n  $u$$_y$=0",
                "left": "left:\\n  $u$$_x$=0",
            }
        """
        labels = {}
        for mesh_name, constraint in self.constraints().items():
            if len(constraint) == 0:
                labels[mesh_name] = mesh_name
                continue
            label = f"{mesh_name}:"

            for process_var, entries in constraint.items():
                for entry in entries:
                    label += self._format_constraint(entry, process_var)
            labels[mesh_name] = label
        return labels
