# https://bbpteam.epfl.ch/project/spaces/display/BGLIB/Neurodamus
# Copyright 2005-2021 Blue Brain Project, EPFL. All rights reserved.
"""Implements applying modifications that mimic experimental manipulations

New Modification classes must be registered, using the appropriate decorator.
Also, when instantiated by the framework, __init__ is passed three arguments
(1) target (2) mod_info: dict (3) cell_manager. Example

>>> @ModificationManager.register_type
>>> class TTX:
>>>
>>> def __init__(self, target, mod_info: dict, cell_manager):
>>>     tpoints = target.get_point_list(cell_manager, section_type, compartment_type)
>>>     for point in tpoints:
>>>         for sec_id, sc in enumerate(point.sclst):
>>>             if not sc.exists():
>>>                 continue
>>>             sec = sc.sec

"""

import ast
import logging

import libsonata

from .core import NeuronWrapper as Nd
from .core.configuration import ConfigurationError
from .target_manager import TargetPointList
from .utils import compat
from .utils.logging import log_verbose


class ModificationManager:
    """A manager for circuit Modifications.
    Overrides HOC manager, as the only Modification there (TTX) is outdated.
    """

    _mod_types = {}  # modification handled in Python

    def __init__(self, target_manager):
        self._target_manager = target_manager
        self._modifications = []

    def interpret(self, target_spec, mod_info):
        mod_t = self._mod_types.get(mod_info.type)

        if not mod_t:
            raise ConfigurationError(f"Unknown Modification {mod_info.type}")
        if isinstance(mod_info, libsonata.SimulationConfig.ModificationCompartmentSet):
            target = self._target_manager.get_compartment_set(target_spec.name)
        else:
            target = self._target_manager.get_target(target_spec)
        cell_manager = self._target_manager._cell_manager
        mod = mod_t(target, mod_info, cell_manager)
        self._modifications.append(mod)

    @classmethod
    def register_type(cls, mod_class):
        cls._mod_types[mod_class.MOD_TYPE] = mod_class
        return mod_class


@ModificationManager.register_type
class TTX:
    """Applies sodium channel block to all sections of the cells in the given target

    Uses TTXDynamicsSwitch as in BGLibPy. Overrides HOC version, which is outdated
    """

    MOD_TYPE = libsonata.SimulationConfig.ModificationBase.ModificationType.ttx

    def __init__(self, target, mod_info: libsonata.SimulationConfig.ModificationTTX, cell_manager):
        tpoints = target.get_point_list(
            cell_manager,
            section_type=libsonata.SimulationConfig.Report.Sections.all,
            compartment_type=libsonata.SimulationConfig.Report.Compartments.all,
        )

        # insert and activate TTX mechanism in all sections of each cell in target
        for tpoint_list in tpoints:
            for sc in tpoint_list.sclst:
                if not sc.exists():  # skip sections not on this split
                    continue
                sec = sc.sec
                if not Nd.ismembrane("TTXDynamicsSwitch", sec=sec):
                    sec.insert("TTXDynamicsSwitch")
                sec.ttxo_level_TTXDynamicsSwitch = 1.0


@ModificationManager.register_type
class ConfigureAllSections:
    """Perform one or more assignments involving section attributes,
    for all sections that have all the referenced attributes.

    Use case is modifying mechanism variables from config.
    """

    MOD_TYPE = libsonata.SimulationConfig.ModificationBase.ModificationType.configure_all_sections

    def __init__(
        self,
        target,
        mod_info: libsonata.SimulationConfig.ModificationConfigureAllSections,
        cell_manager,
    ):
        config, config_attrs = self.parse_section_config(mod_info.section_configure)
        tpoints = target.get_point_list(
            cell_manager,
            section_type=libsonata.SimulationConfig.Report.Sections.all,
            compartment_type=libsonata.SimulationConfig.Report.Compartments.all,
        )

        napply = 0  # number of sections where config applies
        # change mechanism variable in all sections that have it
        for tpoint_list in tpoints:
            for _, sc in enumerate(tpoint_list.sclst):
                if not sc.exists():  # skip sections not on this split
                    continue
                sec = sc.sec
                if all(hasattr(sec, x) for x in config_attrs):  # if has all attributes
                    # unsafe but sanitized
                    exec(config, {"__builtins__": None}, {"sec": sec})  # noqa: S102
                    napply += 1

        log_verbose(f"Applied to {napply} sections")

        if napply == 0:
            logging.warning(
                "configure_all_sections applied to zero sections, "
                "please check its section_configure for possible mistakes"
            )

    def parse_section_config(self, config):
        config = config.replace("%s.", "__sec_wildcard__.")  # wildcard to placeholder
        all_attrs = self.AttributeCollector()
        tree = ast.parse(config)
        for elem in tree.body:  # for each semicolon-separated statement
            # check assignment targets
            for tgt in self.assignment_targets(elem):
                # must be single assignment of a __sec_wildcard__ attribute
                if not isinstance(tgt, ast.Attribute) or tgt.value.id != "__sec_wildcard__":
                    raise ConfigurationError(
                        "section_configure only supports single assignments "
                        "of attributes of the section wildcard %s"
                    )
            all_attrs.visit(elem)  # collect attributes in assignment
        config = config.replace("__sec_wildcard__.", "sec.")  # placeholder to section variable

        return config, all_attrs.attrs

    class AttributeCollector(ast.NodeVisitor):
        """Node visitor collecting all attribute names in a set"""

        attrs = set()

        def visit_Attribute(self, node):
            self.attrs.add(node.attr)

    @staticmethod
    def assignment_targets(node):
        if isinstance(node, ast.Assign):
            return node.targets
        if isinstance(node, ast.AugAssign):
            return [node.target]
        raise ConfigurationError(
            "section_configure must consist of one or more semicolon-separated assignments"
        )


@ModificationManager.register_type
class SectionList:
    """Perform one or more assignments involving section attributes,
    for the sections in the list that have the referenced attributes.

    Use case is modifying mechanism variables from config.
    """

    MOD_TYPE = libsonata.SimulationConfig.ModificationBase.ModificationType.section_list

    def __init__(
        self,
        target,
        mod_info: libsonata.SimulationConfig.ModificationSectionList,
        cell_manager,
    ):
        napply = self.parse_section_config(target, mod_info.section_configure, cell_manager)

        log_verbose(f"Applied to {napply} sections")

        if napply == 0:
            logging.warning(
                "section_list applied to zero sections, "
                "please check its section_configure for possible mistakes"
            )

    def parse_section_config(self, target, config, cell_manager):
        napply = 0
        all_attrs = self.AttributeCollector()
        tree = ast.parse(config)
        for elem in tree.body:  # for each semicolon-separated statement
            # check assignment targets
            for tgt in self.assignment_targets(elem):
                # must be single assignment of a section attribute
                if not isinstance(tgt, ast.Attribute):
                    raise ConfigurationError(
                        "section_configure only supports single assignments "
                        "of attributes of the section"
                    )
            all_attrs.visit(elem)  # collect attributes in assignment

            section = elem.targets[0].value.id
            attr = elem.targets[0].attr
            modif = ast.unparse(elem)
            napply += self.apply_modification(target, section, attr, modif, cell_manager)

        return napply

    @staticmethod
    def apply_modification(target, section, attr, modif, cell_manager):
        if section == "apical":
            section_type = libsonata.SimulationConfig.Report.Sections.apic
        elif section == "axonal":
            section_type = libsonata.SimulationConfig.Report.Sections.axon
        elif section == "basal":
            section_type = libsonata.SimulationConfig.Report.Sections.dend
        elif section == "somatic":
            section_type = libsonata.SimulationConfig.Report.Sections.soma
        elif section == "all":
            section_type = libsonata.SimulationConfig.Report.Sections.all
        else:
            raise ConfigurationError(
                f"Unknown section type: {section}.\n"
                f"Allowed types are: apical, axonal, basal, somatic or all"
            )

        # Filter by section type
        tpoints = target.get_point_list(
            cell_manager,
            section_type=section_type,
            compartment_type=libsonata.SimulationConfig.Report.Compartments.all,
        )

        napply = 0  # number of sections where config applies
        # change mechanism variable in all sections that have it
        for tpoint_list in tpoints:
            for sc in tpoint_list.sclst:
                if not sc.exists():  # skip sections not on this split
                    continue
                sec = sc.sec
                if hasattr(sec, attr):
                    # unsafe but sanitized
                    exec(modif, {"__builtins__": None}, {section: sec})  # noqa: S102
                    napply += 1

        return napply

    class AttributeCollector(ast.NodeVisitor):
        """Node visitor collecting all attribute names in a set"""

        attrs = set()

        def visit_Attribute(self, node):
            self.attrs.add(node.attr)

    @staticmethod
    def assignment_targets(node):
        if isinstance(node, ast.Assign):
            return node.targets
        if isinstance(node, ast.AugAssign):
            return [node.target]
        raise ConfigurationError(
            "section_configure must consist of one or more semicolon-separated assignments"
        )


@ModificationManager.register_type
class Section:
    """Perform one or more assignments involving section attributes,
    for the given section with the referenced attributes.

    Use case is modifying mechanism variables from config.
    """

    MOD_TYPE = libsonata.SimulationConfig.ModificationBase.ModificationType.section

    def __init__(
        self,
        target,
        mod_info: libsonata.SimulationConfig.ModificationSection,
        cell_manager,
    ):
        napply = self.parse_section_config(target, mod_info.section_configure, cell_manager)

        log_verbose(f"Applied to {napply} sections")

        if napply == 0:
            logging.warning(
                "section_list applied to zero sections, "
                "please check its section_configure for possible mistakes"
            )

    def parse_section_config(self, target, config, cell_manager):
        napply = 0
        all_attrs = self.AttributeCollector()
        tree = ast.parse(config)
        for elem in tree.body:  # for each semicolon-separated statement
            # check assignment targets
            for tgt in self.assignment_targets(elem):
                # must be single assignment of a section attribute
                if not isinstance(tgt, ast.Attribute):
                    raise ConfigurationError(
                        "section_configure only supports single assignments "
                        "of attributes of the section"
                    )
            all_attrs.visit(elem)  # collect attributes in assignment

            section, index, attr = self._parse_section_target(elem.targets[0])
            modif = ast.unparse(elem)

            # print(f"section: {section} ; index: {index} ; attr: {attr} ; modif: {modif}")
            napply += self.apply_modification(target, section, index, attr, modif, cell_manager)

        return napply

    @staticmethod
    def _parse_section_target(tgt: ast.Attribute):
        """Extract (section_name, section_index, attribute) from
        targets like soma[0].gnabar_hh
        """
        if not isinstance(tgt.value, ast.Subscript):
            raise ConfigurationError("section must be indexed, e.g. soma[0]")

        sub = tgt.value

        if not isinstance(sub.value, ast.Name):
            raise ConfigurationError("invalid section name")

        if not isinstance(sub.slice, ast.Constant):
            raise ConfigurationError("section index must be constant")

        return sub.value.id, sub.slice.value, tgt.attr

    @staticmethod
    def apply_modification(target, section, idx, attr, modif, cell_manager):
        if section == "apic":
            section_type = libsonata.SimulationConfig.Report.Sections.apic
        elif section == "axon":
            section_type = libsonata.SimulationConfig.Report.Sections.axon
        elif section == "dend":
            section_type = libsonata.SimulationConfig.Report.Sections.dend
        elif section == "soma":
            section_type = libsonata.SimulationConfig.Report.Sections.soma
        else:
            raise ConfigurationError(
                f"Unknown section type: {section}.\nAllowed types are: apic, axon, dend or soma"
            )

        # Filter by section type
        tpoints = target.get_point_list(
            cell_manager,
            section_type=section_type,
            compartment_type=libsonata.SimulationConfig.Report.Compartments.all,
        )

        napply = 0  # number of sections where config applies
        # change mechanism variable in all sections that have it
        for tpoint_list in tpoints:
            if len(tpoint_list.sclst) > idx:
                sec = tpoint_list.sclst[idx].sec
                if hasattr(sec, attr):
                    # print(f"Applying modification: {modif} to section: {sec}")
                    modif_sec = modif.replace(f"{section}[{idx}].", "sec.", 1)
                    # unsafe but sanitized
                    exec(modif_sec, {"__builtins__": None}, {"sec": sec})  # noqa: S102
                    napply += 1

        return napply

    class AttributeCollector(ast.NodeVisitor):
        """Node visitor collecting all attribute names in a set"""

        attrs = set()

        def visit_Attribute(self, node):
            self.attrs.add(node.attr)

    @staticmethod
    def assignment_targets(node):
        if isinstance(node, ast.Assign):
            return node.targets
        if isinstance(node, ast.AugAssign):
            return [node.target]
        raise ConfigurationError(
            "section_configure must consist of one or more semicolon-separated assignments"
        )


@ModificationManager.register_type
class CompartmentSet:
    """Perform one or more assignments involving compartment attributes
    (e.g. cm, hh.gnabar, pas.g) on selected segments from compartment set.

    Use case is modifying mechanism variables from config.
    """

    MOD_TYPE = libsonata.SimulationConfig.ModificationBase.ModificationType.compartment_set

    def __init__(self, target, mod_info, cell_manager):
        napply = self.parse_section_config(target, mod_info.section_configure, cell_manager)

        log_verbose(f"Applied to {napply} segments")

        if napply == 0:
            logging.warning(
                "compartment_set applied to zero segments. Check section_configure for mistakes."
            )

    def parse_section_config(self, target, config, cell_manager):
        napply = 0
        tree = ast.parse(config)

        for stmt in tree.body:
            if not isinstance(stmt, (ast.Assign, ast.AugAssign)):
                raise ConfigurationError("section_configure must contain assignments only")

            targets = self.assignment_targets(stmt)
            if len(targets) != 1:
                raise ConfigurationError("Only single-target assignments are supported")

            lhs_node = targets[0]
            if not isinstance(lhs_node, (ast.Name, ast.Attribute)):
                raise ConfigurationError("Assignments must target variables like cm or hh.gnabar")

            dotted_name = self.get_full_attr_name(lhs_node)
            value = self.evaluate_rhs(stmt.value)
            napply += self.apply_modification(target, dotted_name, value, cell_manager)

        return napply

    def apply_modification(self, target, dotted_name, value, cell_manager):
        napply = 0
        sel_node_set = target.node_ids()

        for cl in target.filtered_iter(sel_node_set):
            raw_gid, section_id, offset = (cl.node_id, cl.section_id, cl.offset)
            cell = cell_manager.get_cell(raw_gid)
            sec = cell.get_sec(section_id)
            seg = sec(offset)

            try:
                self.set_segment_value(seg, dotted_name, value)
                # print(f"Applying comp_set modification: {value} to section: {seg}")
                # print(f"gid: {raw_gid}, section: {section_id}, offset: {offset}")

            except AttributeError:
                continue  # segment doesn't have that mechanism

            napply += 1

        return napply

    @staticmethod
    def get_full_attr_name(node):
        """Reconstruct dotted name from AST Attribute."""
        parts = []

        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value

        if isinstance(node, ast.Name):
            parts.append(node.id)
        else:
            raise ConfigurationError("Unsupported assignment target")

        return ".".join(reversed(parts))

    @staticmethod
    def set_segment_value(seg, dotted_name, value):
        """Resolve dotted attribute on a segment safely."""
        parts = dotted_name.split(".")

        obj = seg
        for p in parts[:-1]:
            obj = getattr(obj, p)

        setattr(obj, parts[-1], value)

    @staticmethod
    def evaluate_rhs(node):
        """Safely evaluate numeric RHS (no eval)."""
        if isinstance(node, ast.Constant):
            return float(node.value)

        if (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant)
        ):
            return -float(node.operand.value)

        raise ConfigurationError("Only numeric constants are allowed in section_configure")

    @staticmethod
    def assignment_targets(node):
        if isinstance(node, ast.Assign):
            return node.targets
        if isinstance(node, ast.AugAssign):
            return [node.target]
        raise ConfigurationError("section_configure must consist of assignments")

    @staticmethod
    def get_point_list_from_compartment_set(
        cell_manager, compartment_set
    ) -> compat.List[TargetPointList]:

        point_list = compat.List()
        sel_node_set = compartment_set.node_ids()

        for cl in compartment_set.filtered_iter(sel_node_set):
            raw_gid, section_id, offset = (cl.node_id, cl.section_id, cl.offset)

            cell = cell_manager.get_cell(raw_gid)
            sec = cell.get_sec(section_id)

            if len(point_list) and point_list[-1].gid == raw_gid:
                point_list[-1].append(section_id, Nd.SectionRef(sec), offset)
            else:
                point = TargetPointList(raw_gid)
                point.append(section_id, Nd.SectionRef(sec), offset)
                point_list.append(point)

        return point_list
