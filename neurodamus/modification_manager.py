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
from collections.abc import Generator

import libsonata

from .cell_distributor import _CellManager
from .core import NeuronWrapper as Nd
from .core.configuration import ConfigurationError
from .metype import BaseCell
from .target_manager import NodesetTarget, TargetSpec
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
        """Interpret a modification entry and apply it to the corresponding target."""
        mod_t = self._mod_types.get(mod_info.type)

        if not mod_t:
            raise ConfigurationError(f"Unknown Modification {mod_info.type}")

        if mod_info.type.name == "compartment_set":
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
        """Register a modification class by its MOD_TYPE."""
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
    def parse_assignments(config: str) -> Generator[tuple[ast.stmt, ast.expr], None, None]:
        """Parse a config string into individual assignment statements and their targets."""
        tree = ast.parse(config)
        for stmt in tree.body:
            targets = BaseASTModification.assignment_targets(stmt)

            if len(targets) != 1:
                raise ConfigurationError(
                    "Only single-target assignments are supported in section_configure"
                )

            yield stmt, targets[0]

    @staticmethod
    def assignment_targets(node: ast.stmt) -> list[ast.expr]:
        """Extract assignment targets from an AST Assign or AugAssign node."""
        if isinstance(node, ast.Assign):
            return node.targets
        if isinstance(node, ast.AugAssign):
            return [node.target]
        raise ConfigurationError("section_configure must contain assignments only")

    @staticmethod
    def evaluate_numeric_rhs(node: ast.expr) -> float:
        """Safely convert numeric right-hand side."""
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


@ModificationManager.register_type
class SectionListModification(BaseASTModification):
    """Perform one or more assignments involving section attributes,
    for the sections in the list that have the referenced attributes.

    Accepted syntax is of the style "apical.gbar_NaTg = 0", with semi-colon-separated assignments

    Use case is modifying mechanism variables from config.
    """

    MOD_TYPE = libsonata.SimulationConfig.ModificationBase.ModificationType.section_list

    def __init__(
        self,
        target: NodesetTarget,
        mod_info: libsonata.SimulationConfig.ModificationSectionList,
        cell_manager: _CellManager,
    ):
        napply = self.apply_config(target, mod_info.section_configure, cell_manager)

        log_verbose(f"Applied to {napply} sections")

        if napply == 0:
            logging.warning(
                "section_list applied to zero sections. Check section_configure for mistakes."
            )

    def apply_config(self, target: NodesetTarget, config: str, cell_manager: _CellManager) -> int:
        """Parse and apply section_list modifications, returns the number of sections modified."""
        napply = 0

        for stmt, lhs in self.parse_assignments(config):
            if not isinstance(lhs, ast.Attribute):
                raise ConfigurationError(
                    "section_list modification must use syntax like apical.gbar = 0"
                )

            if not isinstance(lhs.value, ast.Name):
                raise ConfigurationError("Invalid syntax for section type")

            section_name = lhs.value.id
            attr_name = lhs.attr
            # Validate section type: accept section_list names and "all"
            if section_name != "all" and section_name not in BaseCell._SECTION_LIST_TO_NAME:
                allowed = ", ".join(["all", *BaseCell._SECTION_LIST_TO_NAME])
                raise ConfigurationError(
                    f"Unknown section type: {section_name}. Allowed types are: {allowed}"
                )

            rhs_value = self.evaluate_numeric_rhs(stmt.value)

            for gid in target.get_local_gids():
                cell = cell_manager.get_cellref(gid)
                for sec in getattr(cell, section_name, []):
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
class SectionModification(BaseASTModification):
    """Perform one or more assignments involving section attributes,
    for the given section with the referenced attributes.

    Accepted syntax is of the style "apic[10].gbar_KTst = 0", with semi-colon-separated assignments

    Use case is modifying mechanism variables from config.
    """

    MOD_TYPE = libsonata.SimulationConfig.ModificationBase.ModificationType.section

    def __init__(
        self,
        target: NodesetTarget,
        mod_info: libsonata.SimulationConfig.ModificationSection,
        cell_manager: _CellManager,
    ):
        napply = self.apply_config(target, mod_info.section_configure, cell_manager)

        log_verbose(f"Applied to {napply} sections")

        if napply == 0:
            logging.warning(
                "section applied to zero sections. Check section_configure for mistakes."
            )

    def apply_config(
        self,
        target: NodesetTarget,
        config: str,
        cell_manager: _CellManager,
    ) -> int:
        """Parse and apply section modifications, returns the number of sections modified."""
        napply = 0

        for stmt, lhs in self.parse_assignments(config):
            self.section_sanity_checks(lhs)
            sub = lhs.value
            section_name = sub.value.id
            # Validate section name against known types
            if section_name not in BaseCell._SECTION_NAME_TO_LIST:
                allowed = ", ".join(BaseCell._SECTION_NAME_TO_LIST)
                raise ConfigurationError(
                    f"Unknown section type: {section_name}. Allowed types are: {allowed}"
                )
            idx = sub.slice.value
            attr_name = lhs.attr
            rhs_value = self.evaluate_numeric_rhs(stmt.value)

            for gid in target.get_local_gids():
                cell = cell_manager.get_cellref(gid)
                secs = getattr(cell, section_name, None)
                if secs is None or idx >= len(secs):
                    continue
                sec = secs[idx]
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

    @staticmethod
    def section_sanity_checks(
        lhs: ast.expr,
    ) -> None:
        """Validate that the LHS of an assignment uses correct indexed section syntax."""
        if not isinstance(lhs, ast.Attribute):
            raise ConfigurationError("section modification must use syntax like soma[0].gnabar = 0")

        if not isinstance(lhs.value, ast.Subscript):
            raise ConfigurationError("Section must be indexed")

        sub = lhs.value

        if not isinstance(sub.value, ast.Name):
            raise ConfigurationError("Invalid syntax for section type")

        if not isinstance(sub.slice, ast.Constant):
            raise ConfigurationError("Section index must be constant")


@ModificationManager.register_type
class CompartmentSetModification(BaseASTModification):
    """Perform one or more assignments involving compartment attributes on selected segments from
    compartment set.

    Accepted syntax is of the style "gbar_Ca_HVA2 = 1.5", with semi-colon-separated assignments

    Use case is modifying mechanism variables from config.
    """

    MOD_TYPE = libsonata.SimulationConfig.ModificationBase.ModificationType.compartment_set

    def __init__(
        self,
        target: libsonata.CompartmentSet,
        mod_info: libsonata.SimulationConfig.ModificationCompartmentSet,
        cell_manager: _CellManager,
    ):
        napply = self.apply_config(target, mod_info.section_configure, cell_manager)

        log_verbose(f"Applied to {napply} segments")

        if napply == 0:
            logging.warning(
                "compartment_set applied to zero segments. Check section_configure for mistakes."
            )

    def apply_config(
        self,
        target: libsonata.CompartmentSet,
        config: str,
        cell_manager: _CellManager,
    ) -> int:
        """Parse and apply compartment_set modif, returns the number of segments modified."""
        napply = 0

        for stmt, lhs in self.parse_assignments(config):
            if not isinstance(lhs, ast.Name):
                raise ConfigurationError(
                    "compartment_set modification must target variables like 'gnabar_hh'"
                )

            comp_attr = lhs.id
            rhs_value = self.evaluate_numeric_rhs(stmt.value)

            local_gids = cell_manager.get_final_gids()

            for cl in target.filtered_iter(target.node_ids()):
                if cl.node_id not in local_gids:
                    continue
                cell = cell_manager.get_cell(cl.node_id)
                sec = cell.get_sec(cl.section_id)
                seg = sec(cl.offset)

                if not hasattr(seg, comp_attr):
                    continue

                # Treat ast.Assign as default
                # parse_assignments already checked it is either ast.Assign or ast.AugAssign
                new_value = rhs_value

                if isinstance(stmt, ast.AugAssign):
                    current = getattr(seg, comp_attr)
                    op_type = type(stmt.op)

                    if op_type not in BaseASTModification.AUG_OPS:
                        raise ConfigurationError(f"Unsupported operator {op_type}")

                    new_value = BaseASTModification.AUG_OPS[op_type](current, rhs_value)

                setattr(seg, comp_attr, new_value)
                napply += 1

        return napply
