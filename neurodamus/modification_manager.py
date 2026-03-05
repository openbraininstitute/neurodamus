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
import operator

import libsonata

from .core import NeuronWrapper as Nd
from .core.configuration import ConfigurationError
from .target_manager import TargetSpec
from .utils.logging import log_verbose


class ModificationManager:
    """A manager for circuit Modifications.
    Overrides HOC manager, as the only Modification there (TTX) is outdated.
    """

    _mod_types = {}  # modification handled in Python

    def __init__(self, target_manager):
        self._target_manager = target_manager
        self._modifications = []

    def interpret(self, mod_info):
        mod_t = self._mod_types.get(mod_info.type)

        if not mod_t:
            raise ConfigurationError(f"Unknown Modification {mod_info.type}")

        if isinstance(mod_info, libsonata.SimulationConfig.ModificationCompartmentSet):
            target_spec = TargetSpec(mod_info.compartment_set, None)
            target = self._target_manager.get_compartment_set(target_spec.name)
        else:
            target_spec = TargetSpec(mod_info.node_set, None)
            target = self._target_manager.get_target(target_spec)

        logging.info(" * [MOD] %s: %s -> %s", mod_info.name, mod_info.type.name, target_spec)

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


class BaseASTModification:
    """Common AST parsing helpers for assignment-based modification configuration."""

    AUG_OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }

    @staticmethod
    def parse_assignments(config: str):
        tree = ast.parse(config)
        for stmt in tree.body:
            targets = BaseASTModification.assignment_targets(stmt)

            if len(targets) != 1:
                raise ConfigurationError(
                    "Only single-target assignments are supported in section_configure"
                )

            yield stmt, targets[0]

    @staticmethod
    def assignment_targets(node):
        if isinstance(node, ast.Assign):
            return node.targets
        if isinstance(node, ast.AugAssign):
            return [node.target]
        raise ConfigurationError("section_configure must contain assignments only")

    @staticmethod
    def evaluate_numeric_rhs(node):
        """Safely evaluate numeric right-hand side (no evaluation)."""
        # Positive constants
        if isinstance(node, ast.Constant):
            return float(node.value)

        # Negative constants
        if (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant)
        ):
            return -float(node.operand.value)

        raise ConfigurationError("Only numeric constants are allowed in section_configure")

    @staticmethod
    def resolve_dotted_attr(obj, dotted_name):
        parts = dotted_name.split(".")
        for p in parts[:-1]:
            if hasattr(obj, p):
                obj = getattr(obj, p)
            else:
                return obj, None
        return obj, parts[-1]


class BaseSectionModification(BaseASTModification):
    """Shared base class for section and section_list modifications."""

    SECTION_MAP = {
        # section_list type entries
        "apical": libsonata.SimulationConfig.Report.Sections.apic,
        "axonal": libsonata.SimulationConfig.Report.Sections.axon,
        "basal": libsonata.SimulationConfig.Report.Sections.dend,
        "somatic": libsonata.SimulationConfig.Report.Sections.soma,
        "all": libsonata.SimulationConfig.Report.Sections.all,
        # section type entries
        "apic": libsonata.SimulationConfig.Report.Sections.apic,
        "axon": libsonata.SimulationConfig.Report.Sections.axon,
        "dend": libsonata.SimulationConfig.Report.Sections.dend,
        "soma": libsonata.SimulationConfig.Report.Sections.soma,
    }

    def get_allowed_entries(self):
        return ", ".join(sorted(self.SECTION_MAP))

    def get_section_type(self, name: str):
        try:
            return self.SECTION_MAP[name]
        except KeyError as err:
            allowed = self.get_allowed_entries()
            raise ConfigurationError(
                f"Unknown section type: {name}. Allowed types are: {allowed}"
            ) from err

    @staticmethod
    def iter_sections(target, cell_manager, section_type):
        tpoints = target.get_point_list(
            cell_manager,
            section_type=section_type,
            compartment_type=libsonata.SimulationConfig.Report.Compartments.all,
        )

        sections = set()
        for tpoint_list in tpoints:
            for sc in tpoint_list.sclst:
                if sc.exists():
                    sections.add(sc.sec)

        return sections


@ModificationManager.register_type
class SectionList(BaseSectionModification):
    """Perform one or more assignments involving section attributes,
    for the sections in the list that have the referenced attributes.

    Accepted syntax is of the style "apical.gbar_NaTg = 0", with semi-colon-separated assignments

    Use case is modifying mechanism variables from config.
    """

    SECTION_MAP = {
        "apical": libsonata.SimulationConfig.Report.Sections.apic,
        "axonal": libsonata.SimulationConfig.Report.Sections.axon,
        "basal": libsonata.SimulationConfig.Report.Sections.dend,
        "somatic": libsonata.SimulationConfig.Report.Sections.soma,
        "all": libsonata.SimulationConfig.Report.Sections.all,
    }

    MOD_TYPE = libsonata.SimulationConfig.ModificationBase.ModificationType.section_list

    def __init__(self, target, mod_info, cell_manager):
        napply = self.apply_config(target, mod_info.section_configure, cell_manager)

        log_verbose(f"Applied to {napply} sections")

        if napply == 0:
            logging.warning(
                "section_list applied to zero sections. Check section_configure for mistakes."
            )

    def apply_config(self, target, config, cell_manager):
        napply = 0

        for stmt, lhs in self.parse_assignments(config):
            if not isinstance(lhs, ast.Attribute):
                raise ConfigurationError(
                    "section_list modification must use syntax like apical.gbar = 0"
                )

            if not isinstance(lhs.value, ast.Name):
                raise ConfigurationError("Invalid section target")

            section_name = lhs.value.id
            attr_name = lhs.attr
            section_type = self.get_section_type(section_name)

            rhs_value = self.evaluate_numeric_rhs(stmt.value)

            for sec in self.iter_sections(target, cell_manager, section_type):
                if not hasattr(sec, attr_name):
                    continue

                # Treat ast.Assign as default
                # parse_assignments already checked it is either ast.Assign or ast.AugAssign
                new_value = rhs_value

                if isinstance(stmt, ast.AugAssign):
                    current = getattr(sec, attr_name)
                    op_type = type(stmt.op)

                    if op_type not in BaseASTModification.AUG_OPS:
                        raise ConfigurationError(f"Unsupported operator {op_type}")

                    new_value = BaseASTModification.AUG_OPS[op_type](current, rhs_value)

                setattr(sec, attr_name, new_value)
                napply += 1

        return napply


@ModificationManager.register_type
class Section(BaseSectionModification):
    """Perform one or more assignments involving section attributes,
    for the given section with the referenced attributes.

    Accepted syntax is of the style "apic[10].gbar_KTst = 0", with semi-colon-separated assignments

    Use case is modifying mechanism variables from config.
    """

    SECTION_MAP = {
        "apic": libsonata.SimulationConfig.Report.Sections.apic,
        "axon": libsonata.SimulationConfig.Report.Sections.axon,
        "dend": libsonata.SimulationConfig.Report.Sections.dend,
        "soma": libsonata.SimulationConfig.Report.Sections.soma,
    }

    MOD_TYPE = libsonata.SimulationConfig.ModificationBase.ModificationType.section

    def __init__(self, target, mod_info, cell_manager):
        napply = self.apply_config(target, mod_info.section_configure, cell_manager)

        log_verbose(f"Applied to {napply} sections")

        if napply == 0:
            logging.warning(
                "section applied to zero sections. Check section_configure for mistakes."
            )

    def apply_config(self, target, config, cell_manager):
        napply = 0

        for stmt, lhs in self.parse_assignments(config):
            self.section_sanity_checks(lhs)
            sub = lhs.value
            section_name = sub.value.id
            idx = sub.slice.value
            attr_name = lhs.attr
            rhs_value = self.evaluate_numeric_rhs(stmt.value)

            target_cells = self.get_target_cells(target, cell_manager, section_name)

            for cell in target_cells:
                sec_list = getattr(cell, section_name)
                if len(sec_list) <= idx:
                    raise ValueError(f"{idx} array index out of range (length = {len(sec_list)})")

                sec = sec_list[idx]

                if not hasattr(sec, attr_name):
                    continue

                # Treat ast.Assign as default
                # parse_assignments already checked it is either ast.Assign or ast.AugAssign
                new_value = rhs_value

                if isinstance(stmt, ast.AugAssign):
                    current = getattr(sec, attr_name)
                    op_type = type(stmt.op)

                    if op_type not in BaseASTModification.AUG_OPS:
                        raise ConfigurationError(f"Unsupported operator {op_type}")

                    new_value = BaseASTModification.AUG_OPS[op_type](current, rhs_value)

                setattr(sec, attr_name, new_value)
                napply += 1

        return napply

    def get_target_cells(self, target, cell_manager, section_name):
        section_type = self.get_section_type(section_name)

        tpoints = target.get_point_list(
            cell_manager,
            section_type=section_type,
            compartment_type=libsonata.SimulationConfig.Report.Compartments.all,
        )

        target_cells = set()
        for tpoint_list in tpoints:
            for sec in tpoint_list.sclst:
                cell = sec.sec.cell()
                target_cells.add(cell)
                break

        return target_cells

    @staticmethod
    def section_sanity_checks(lhs):
        if not isinstance(lhs, ast.Attribute):
            raise ConfigurationError("section modification must use syntax like soma[0].gnabar = 0")

        if not isinstance(lhs.value, ast.Subscript):
            raise ConfigurationError("Section must be indexed")

        sub = lhs.value

        if not isinstance(sub.value, ast.Name):
            raise ConfigurationError("Invalid section name")

        if not isinstance(sub.slice, ast.Constant):
            raise ConfigurationError("Section index must be constant")


@ModificationManager.register_type
class CompartmentSet(BaseASTModification):
    """Perform one or more assignments involving compartment attributes on selected segments from
    compartment set.

    Accepted syntax is of the style "gbar_Ca_HVA2 = 1.5", with semi-colon-separated assignments

    Use case is modifying mechanism variables from config.
    """

    MOD_TYPE = libsonata.SimulationConfig.ModificationBase.ModificationType.compartment_set

    def __init__(self, target, mod_info, cell_manager):
        napply = self.apply_config(target, mod_info.section_configure, cell_manager)

        log_verbose(f"Applied to {napply} segments")

        if napply == 0:
            logging.warning(
                "compartment_set applied to zero segments. Check section_configure for mistakes."
            )

    def apply_config(self, target, config, cell_manager):
        napply = 0

        for stmt, lhs in self.parse_assignments(config):
            if not isinstance(lhs, (ast.Name, ast.Attribute)):
                raise ConfigurationError(
                    "compartment_set modification must target properties like 'hh.gnabar' or "
                    "'gnabar_hh'"
                )

            dotted_name = self.get_full_attr_name(lhs)
            rhs_value = self.evaluate_numeric_rhs(stmt.value)

            for cl in target.filtered_iter(target.node_ids()):
                cell = cell_manager.get_cell(cl.node_id)
                sec = cell.get_sec(cl.section_id)
                seg = sec(cl.offset)

                obj, final_attr = self.resolve_dotted_attr(seg, dotted_name)

                if final_attr is None or not hasattr(obj, final_attr):
                    continue

                # Treat ast.Assign as default
                # parse_assignments already checked it is either ast.Assign or ast.AugAssign
                new_value = rhs_value

                if isinstance(stmt, ast.AugAssign):
                    current = getattr(obj, final_attr)
                    op_type = type(stmt.op)

                    if op_type not in BaseASTModification.AUG_OPS:
                        raise ConfigurationError(f"Unsupported operator {op_type}")

                    new_value = BaseASTModification.AUG_OPS[op_type](current, rhs_value)

                setattr(obj, final_attr, new_value)
                napply += 1

        return napply

    @staticmethod
    def get_full_attr_name(node):
        parts = []

        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value

        if isinstance(node, ast.Name):
            parts.append(node.id)
        else:
            raise ConfigurationError("Unsupported assignment target")

        return ".".join(reversed(parts))
