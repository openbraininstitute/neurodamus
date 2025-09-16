import itertools
import logging
from collections import defaultdict
from collections.abc import Iterator
from functools import lru_cache

import libsonata
import numpy as np

from .core import NeuronWrapper as Nd
from .core.configuration import ConfigurationError
from .core.nodeset import SelectionNodeSet
from .report_parameters import CompartmentType, SectionType
from .utils import compat
from .utils.logging import log_verbose


class TargetError(Exception):
    """A Exception class specific to data error with targets and nodesets"""


class TargetSpec:
    """Definition of a new-style target, accounting for multipopulation"""

    GLOBAL_TARGET_NAME = "_ALL_"

    def __init__(self, target_name):
        """Initialize a target specification

        Args:
            target_name: the target name. For specifying a population use
                the format ``population:target_name``
        """
        if target_name and ":" in target_name:
            self.population, self.name = target_name.split(":")
        else:
            self.name = target_name
            self.population = None
        if not self.name:
            self.name = None

    def __str__(self):
        return (
            (self.name or "")
            if self.population is None
            else "{}:{}".format(self.population, self.name or "")
        )

    def __repr__(self):
        return "<TargetSpec: " + str(self) + ">"

    @property
    def simple_name(self):
        if self.name is None and self.population is None:
            return self.GLOBAL_TARGET_NAME
        return str(self).replace(":", "_")

    def disjoint_populations(self, other):
        # When a population is None we cannot draw conclusions
        #  - In Sonata there's no filtering and target may have multiple
        if self.population is None or other.population is None:
            return False
        # We are only sure if both are specified and different
        return self.population != other.population

    def overlap_byname(self, other):
        return not self.name or not other.name or self.name == other.name

    def overlap(self, other):
        """Are these target specs bound to overlap?
        If not, they still might be overlap, but Target gids need to be inspected
        """
        return self.population == other.population and self.overlap_byname(other)

    def __eq__(self, other):
        return other.population == self.population and other.name == self.name

    __hash__ = None


class TargetManager:
    def __init__(self, run_conf):
        """Initializes a new TargetManager"""
        self._run_conf = run_conf
        self._cell_manager = None
        self._targets = {}
        self._section_access = {}
        self._nodeset_reader = self._init_nodesets(run_conf)
        self._compartment_sets = self._init_compartment_sets(run_conf)
        # A list of the local node sets
        self.local_nodes = []

    @classmethod
    def _init_nodesets(cls, run_conf):
        config_nodeset_file = run_conf.get("config_node_sets_file", None)
        simulation_nodesets_file = run_conf.get("node_sets_file")
        if not simulation_nodesets_file and "TargetFile" in run_conf:
            target_file = run_conf["TargetFile"]
            if target_file.endswith(".json"):
                simulation_nodesets_file = target_file
        return (config_nodeset_file or simulation_nodesets_file) and NodeSetReader(
            config_nodeset_file, simulation_nodesets_file
        )

    @staticmethod
    def _init_compartment_sets(run_conf):
        return (
            libsonata.CompartmentSets.from_file(run_conf["compartment_sets_file"])
            if run_conf["compartment_sets_file"]
            else {}
        )

    def load_targets(self, circuit):
        """Provided that the circuit location is known and whether a nodes file has been
        specified, load any target files.
        Note that these will be moved into a TargetManager after the cells have been distributed,
        instantiated, and potentially split.
        """

        def _is_sonata_file(file_name):
            return file_name.endswith(".h5")

        nodes_file = circuit.get("CellLibraryFile")
        if nodes_file and _is_sonata_file(nodes_file) and self._nodeset_reader:
            self._nodeset_reader.register_node_file(nodes_file)

    @classmethod
    def create_global_target(cls):
        return NodesetTarget(TargetSpec.GLOBAL_TARGET_NAME, [])

    def register_cell_manager(self, cell_manager):
        self._cell_manager = cell_manager

    def register_target(self, target):
        self._targets[target.name] = target

    def register_local_nodes(self, local_nodes):
        """Registers the local nodes so that targets can be scoped to current rank"""
        self.local_nodes.append(local_nodes)

    def clear_simulation_data(self):
        self.local_nodes.clear()
        # deference SectionRefs to sections
        self._section_access.clear()

    def get_compartment_set(self, name: str):
        if self._compartment_sets is None or name not in self._compartment_sets:
            raise ConfigurationError(
                f"Unknown compartment set: {name}. "
                f"Known compartment sets: {', '.join(self._compartment_sets)}"
            )
        return self._compartment_sets[name]

    def get_target(self, target_spec: TargetSpec, target_pop=None):
        """Retrieves a target from any .target file or Sonata nodeset files.

        Targets are generic groups of cells not necessarily restricted to a population.
        When retrieved from the source files they can be cached.
        Targets retrieved from Sonata nodesets keep a reference to all Sonata
        node datasets and can be asked for a sub-target of a specific population.
        """
        if not isinstance(target_spec, TargetSpec):
            target_spec = TargetSpec(target_spec)
        if target_pop:
            target_spec.population = target_pop
        target_name = target_spec.name or TargetSpec.GLOBAL_TARGET_NAME
        target_pop = target_spec.population

        def get_concrete_target(target):
            """Get a more specific target, depending on specified population prefix"""
            target.update_local_nodes(self.local_nodes)
            return target if target_pop is None else target.make_subtarget(target_pop)

        # Check cached
        if target_name in self._targets:
            target = self._targets[target_name]
            return get_concrete_target(target)

        # Check if we can get a Nodeset
        target = self._nodeset_reader and self._nodeset_reader.read_nodeset(target_name)
        if target is not None:
            log_verbose("Retrieved `%s` from Sonata nodeset", target_spec)
            self.register_target(target)
            return get_concrete_target(target)

        raise ConfigurationError(f"Target {target_name} can't be loaded. Check target sources")

    @lru_cache  # noqa: B019
    def intersecting(self, target1, target2):
        """Checks whether two targets intersect"""
        target1_spec = TargetSpec(target1)
        target2_spec = TargetSpec(target2)
        if target1_spec.disjoint_populations(target2_spec):
            return False
        if target1_spec.overlap(target2_spec):
            return True

        # Couldn't get any conclusion from bare target spec
        # Obtain the targets to analyze
        t1, t2 = self.get_target(target1_spec), self.get_target(target2_spec)

        # Check for Sonata nodesets, they might have the same population and overlap
        if set(t1.populations) == set(t2.populations) and target1_spec.overlap_byname(target2_spec):
            return True

        # TODO: Investigate this might yield different results depending on the rank.
        return t1.intersects(t2)  # Otherwise go with full gid intersection

    def pathways_overlap(self, conn1, conn2, equal_only=False):
        src1, dst1 = conn1["Source"], conn1["Destination"]
        src2, dst2 = conn2["Source"], conn2["Destination"]
        if equal_only:
            return TargetSpec(src1) == TargetSpec(src2) and TargetSpec(dst1) == TargetSpec(dst2)
        return self.intersecting(src1, src2) and self.intersecting(dst1, dst2)

    def gid_to_sections(self, gid):
        """For a given gid, return a list of section references stored for random access.
        If the list does not exist, it is built and stored for future use.

        :param gid: GID of the cell
        :return: SerializedSections object with SectionRefs to every section in the cell
        """
        if gid not in self._section_access:
            cell_ref = self._cell_manager.get_cellref(gid)
            if cell_ref is None:
                logging.warning("No cell found for GID: %d", gid)
                return None
            result_serial = SerializedSections(cell_ref)
            self._section_access[gid] = result_serial
        else:
            result_serial = self._section_access[gid]

        return result_serial

    def location_to_point(self, gid, isec, ipt, offset):
        """Given a location for a cell, section id, segment id, and offset into the segment,
        create a list containing a section reference to there.

        :param gid: GID of the cell
        :param isec: Section index
        :param ipt: Distance to start of segment
        :param offset: Offset distance beyond the ipt (microns)
        :return: List with 1 item, where the synapse should go
        """
        # Soma connection, just zero it
        offset = max(offset, 0)

        result_point = TPointList(gid)
        cell_sections = self.gid_to_sections(gid)
        if not cell_sections:
            raise Exception("Getting locations for non-bg sims is not implemented yet...")

        if isec >= cell_sections.num_sections:
            raise Exception(
                f"Error: section {isec} out of bounds ({cell_sections.num_sections} "
                "total). Morphology section count is low, is this a good morphology?"
            )

        distance = 0.5
        tmp_section = cell_sections.isec2sec[int(isec)]

        if tmp_section is None:  # Assume we are in LoadBalance mode
            result_point.append(-1, None, -1)
        elif ipt == -1:
            # Sonata spec have a pre-calculated distance field.
            # In such cases, segment (ipt) is -1 and offset is that distance.
            offset = max(min(offset, 0.9999999), 0.0000001)
            result_point.append(int(isec), tmp_section, offset)
        else:
            # Adjust for section orientation and calculate distance
            section = tmp_section.sec
            if section.orientation() == 1:
                ipt = section.n3d() - 1 - ipt
                offset = -offset

            if ipt < section.n3d():
                distance = (section.arc3d(int(ipt)) + offset) / section.L
                distance = max(min(distance, 0.9999999), 0.0000001)

            if section.orientation() == 1:
                distance = 1 - distance

            result_point.append(int(isec), tmp_section, distance)
        return result_point


class NodeSetReader:
    """Implements reading Sonata Nodesets"""

    def __init__(self, config_nodeset_file, simulation_nodesets_file):
        def _load_nodesets_from_file(nodeset_file):
            if not nodeset_file:
                return libsonata.NodeSets("{}")
            return libsonata.NodeSets.from_file(nodeset_file)

        self._population_stores = {}
        self.nodesets = _load_nodesets_from_file(config_nodeset_file)
        simulation_nodesets = _load_nodesets_from_file(simulation_nodesets_file)
        duplicate_nodesets = self.nodesets.update(simulation_nodesets)
        if duplicate_nodesets:
            logging.warning("Some node set rules were replaced from %s", simulation_nodesets_file)

    def register_node_file(self, node_file):
        storage = libsonata.NodeStorage(node_file)
        for pop_name in storage.population_names:
            self._population_stores[pop_name] = storage

    def __contains__(self, nodeset_name):
        return nodeset_name in self.nodesets.names

    @property
    def names(self):
        return self.nodesets.names

    def read_nodeset(self, nodeset_name: str):
        """Build node sets capable of offsetting.
        The empty population has a special meaning in Sonata, it matches
        all populations in simulation
        """
        if nodeset_name not in self.nodesets.names:
            return None

        def _get_nodeset(pop_name):
            storage = self._population_stores.get(pop_name)
            population = storage.open_population(pop_name)
            # Create SelectionNodeSet object with 1-based gids
            try:
                node_selection = self.nodesets.materialize(nodeset_name, population)
            except libsonata.SonataError as e:
                msg = (
                    f"SonataError for nodeset {nodeset_name} "
                    f'from population "{pop_name}" : {e!s}, skip'
                )
                logging.warning(msg)
                return None
            if node_selection:
                logging.debug("Nodeset %s: Appending gids from %s", nodeset_name, pop_name)
                ns = SelectionNodeSet.from_zero_based_libsonata_selection(node_selection)
                ns.register_global(pop_name)
                return ns
            return None

        nodesets = (_get_nodeset(pop_name) for pop_name in self._population_stores)
        nodesets = [ns for ns in nodesets if ns]
        return NodesetTarget(nodeset_name, nodesets)


class NodesetTarget:
    """Represents a subset of nodes defined in a node_sets.json file.

    A `NodesetTarget` is constructed based on key-value pairs from the `node_sets.json` file,
    where each key defines an object containing a group of nodes. Nodes in a target can be
    selected from one or more populations, and they are internally organized into `nodesets`
    based on their respective populations.

    Example:
    Given populations:
    ```python
    populations = {
    "pop_A": [0, 1, 2],
    "pop_B": [1000, 1001],
    "pop_C": [2000, 2001, 2002]
    }
    ```
    A target could look like:
    ```python
    target = [0, 1, 1000, 1001]
    ```
    Internally, `NodesetTarget` would organize these nodes into:
    ```python
    nodesets = [
    SelectionNodeSet([0, 1]),
    SelectionNodeSet([1000, 1001])
    ]
    ```
    """

    def __init__(self, name, nodesets: list[SelectionNodeSet], local_nodes=None, **_kw):
        self.name = name
        self.nodesets = nodesets
        self.local_nodes = local_nodes

    def __repr__(self):
        nodesets_str = "\n  ".join(str(ns) for ns in self.nodesets)
        return f"NodesetTarget(name={self.name!r}, nodesets=[\n  {nodesets_str}\n])"

    def gid_count(self):
        """Total number of nodes"""
        return sum(len(ns) for ns in self.nodesets)

    def max_gid_count_per_population(self):
        """Max number of nodes per population

        Useful to distribute in multicycle runs
        """
        return max(len(ns) for ns in self.nodesets)

    def get_gids(self):
        """Retrieve the final gids of the nodeset target"""
        if not self.nodesets:
            logging.warning("Nodeset '%s' can't be materialized. No node populations", self.name)
            return np.array([])
        nodesets = sorted(self.nodesets, key=lambda n: n.offset)  # Get gids ascending
        gids = nodesets[0].gids(raw_gids=False)
        for extra_nodes in nodesets[1:]:
            gids = np.append(gids, extra_nodes.gids(raw_gids=False))
        return gids

    def get_raw_gids(self):
        """Retrieve the raw gids of the nodeset target"""
        if not self.nodesets:
            logging.warning("Nodeset '%s' can't be materialized. No node populations", self.name)
            return []
        if len(self.nodesets) > 1:
            raise TargetError("Can not get raw gids for Nodeset target with multiple populations.")
        return np.array(self.nodesets[0].gids(raw_gids=True))

    def __contains__(self, gid):
        """Determine if a given gid is included in the gid list for this target regardless of rank.

        Offsetting is taken into account
        """
        return self.contains(gid)

    def append_nodeset(self, nodeset: SelectionNodeSet):
        """Add a nodeset to the current target"""
        self.nodesets.append(nodeset)

    @property
    def population_names(self):
        return {ns.population_name for ns in self.nodesets}

    @property
    def populations(self):
        return {ns.population_name: ns for ns in self.nodesets}

    def make_subtarget(self, pop_name):
        """A nodeset subtarget contains only one given population"""
        nodesets = [ns for ns in self.nodesets if ns.population_name == pop_name]
        local_nodes = [n for n in self.local_nodes if n.population_name == pop_name]
        return NodesetTarget(f"{self.name}#{pop_name}", nodesets, local_nodes)

    def is_void(self):
        return len(self.nodesets) == 0

    def update_local_nodes(self, local_nodes):
        """Allows setting the local gids"""
        self.local_nodes = local_nodes

    def get_local_gids(self, raw_gids=False):
        """Return the list of target gids in this rank (with offset)"""
        assert self.local_nodes, "Local nodes not set"

        def pop_gid_intersect(nodeset: SelectionNodeSet, raw_gids=False):
            for local_ns in self.local_nodes:
                if local_ns.population_name == nodeset.population_name:
                    return nodeset.intersection(local_ns, raw_gids)
            return []

        if raw_gids:
            assert len(self.nodesets) == 1, "Multiple populations when asking for raw gids"
            return pop_gid_intersect(self.nodesets[0], raw_gids=True)

        gids_groups = tuple(pop_gid_intersect(ns) for ns in self.nodesets)

        return np.concatenate(gids_groups) if gids_groups else np.empty(0, dtype=np.uint32)

    def get_point_list_from_compartment_set(self, cell_manager, compartment_set):
        """Builds a list of points grouped by GID from a compartment set,
        mapping sections and offsets for each relevant population.

        Note:
            The compartment locations in `compartment_set` are expected to be
            sorted and without duplicates. This is ensured by libsonata.
        """
        point_list = compat.List()
        population_name = compartment_set.population

        if population_name not in self.population_names:
            return point_list
        sel_node_set = self.populations[population_name]

        # compartment_set is 0-based
        for cl in compartment_set.filtered_iter(sel_node_set.get_selection(offset=-1)):
            raw_gid, section_id, offset = cl.node_id, cl.section_id, cl.offset
            # points are 1-based
            gid = sel_node_set._offset + raw_gid + 1
            cell = cell_manager.get_cell(gid)
            sec = cell.get_sec(section_id)
            if len(point_list) and point_list[-1].gid == gid:
                point_list[-1].append(section_id, Nd.SectionRef(sec), offset)
            else:
                point = TPointList(gid)
                point.append(section_id, Nd.SectionRef(sec), offset)
                point_list.append(point)

        return point_list

    def get_point_list(
        self,
        cell_manager,
        section_type: SectionType = SectionType.SOMA,
        compartment_type: CompartmentType = CompartmentType.CENTER,
    ):
        """Retrieve a TPointList containing compartments (based on section type and
        compartment type) of any local cells on the cpu.

        Args:
            cell_manager: a cell manager or global cell manager
            section_type: SectionType
            compartment_type: CompartmentType

        Returns:
            list of TPointList containing the compartment position and retrieved section references
        """
        section_type_str = section_type.to_string()
        point_lists = compat.List()

        for gid in self.get_local_gids():
            point_list = TPointList(gid)
            cell = cell_manager.get_cell(gid)
            secs = getattr(cell.CellRef, section_type_str)

            for sec in secs:
                section_id = cell.get_section_id(sec)
                if compartment_type == CompartmentType.CENTER:
                    point_list.append(section_id, Nd.SectionRef(sec), 0.5)
                else:
                    for seg in sec:
                        point_list.append(section_id, Nd.SectionRef(sec), seg.x)
            point_lists.append(point_list)

        return point_lists

    def generate_subtargets(self, n_parts):
        """Generate sub NodeSetTarget per population for multi-cycle runs
        Returns:
            list of [sub_target_n_pop1, sub_target_n_pop2, ...]
        """
        if not n_parts or n_parts == 1:
            return False

        all_raw_gids = {ns.population_name: ns.gids(raw_gids=True) for ns in self.nodesets}

        new_targets = defaultdict(list)
        pop_names = list(all_raw_gids.keys())

        for cycle_i in range(n_parts):
            for pop in pop_names:
                # name sub target per populaton, to be registered later
                target_name = f"{pop}__{self.name}_{cycle_i}"
                target = NodesetTarget(target_name, [SelectionNodeSet().register_global(pop)])
                new_targets[pop].append(target)

        for pop, raw_gids in all_raw_gids.items():
            target_looper = itertools.cycle(new_targets[pop])
            for gid in raw_gids:
                target = next(target_looper)
                target.nodesets[0].add_gids([gid])

        # return list of subtargets lists of all pops per cycle
        return [
            [targets[cycle_i] for targets in new_targets.values()] for cycle_i in range(n_parts)
        ]

    def contains(self, items, raw_gids=False):
        """Return a bool or an array of bool's whether the elements are contained"""
        # Shortcut for empty target. Algorithm below would fail
        if not self.gid_count():
            return ([False] * len(items)) if hasattr(items, "__len__") else False

        gids = self.get_raw_gids() if raw_gids else self.get_gids()
        contained = np.isin(items, gids, kind="table")
        return bool(contained) if contained.ndim == 0 else contained

    def intersects(self, other):
        """Check if two targets intersect. At least one common population has to intersect"""
        if self.population_names.isdisjoint(other.population_names):
            return False

        other_pops = other.populations  # may be created on the fly
        # We loop over one target populations and check the other existence and intersection
        for pop, nodeset in self.populations.items():
            if pop not in other_pops:
                continue
            if nodeset.intersects(other_pops[pop]):
                return True
        return False


class SerializedSections:
    """Serializes the sections of a cell for easier random access.
    Note that this is possible because the v field in the section has been assigned
    an integer corresponding to the target index as read from the morphology file.
    """

    def __init__(self, cell):
        self.num_sections = int(cell.nSecAll)
        # Initialize list to store SectionRef objects
        self.isec2sec = [None] * self.num_sections
        # Flag to control warning message display
        self._serialized_sections_warned = False

        for index, sec in enumerate(cell.all):
            # Accessing the 'v' value at location 0.0001 of the section
            v_value = sec(0.0001).v
            if v_value >= self.num_sections:
                logging.debug("{%s} v(1)={sec(1).v} n3d()={%f}", sec.name(), sec.n3d())
                raise Exception("Error: failure in mk2_isec2sec()")
            if v_value < 0:
                if not self._serialized_sections_warned:
                    logging.warning(
                        "SerializedSections: v(0.0001) < 0. index={%d} v()={%f}",
                        index,
                        v_value,
                    )
                    self._serialized_sections_warned = True
            else:
                # Store a SectionRef to the section at the index specified by v_value
                self.isec2sec[int(v_value)] = Nd.SectionRef(sec=sec)


class TPointList:
    def __init__(self, gid: int):
        self.gid: int = gid
        self.sclst_ids: list = []  # List of section ids
        self.sclst: list = []  # List of section references
        self.x: list = []  # List of point values

    def append(self, section_id: int, section: object, point: float) -> None:
        self.x.append(point)
        self.sclst.append(section)
        self.sclst_ids.append(section_id)

    def extend(self, other: "TPointList") -> None:
        assert isinstance(other, TPointList), f"Expected TPointList, got {type(other).__name__}"
        self.x.extend(other.x)
        self.sclst.extend(other.sclst)
        self.sclst_ids.extend(other.sclst_ids)

    def validate(self) -> None:
        if len(self.x) != len(self.sclst):
            raise RuntimeError(
                f"TPointList invariant violated: x has {len(self.x)} elements, "
                f"sclst has {len(self.sclst)} elements. Expected both lists to be of equal length."
            )

    def __len__(self) -> int:
        self.validate()
        return len(self.sclst)

    def __iter__(self) -> Iterator[tuple[int, object, float]]:
        self.validate()
        return iter(zip(self.sclst_ids, self.sclst, self.x))

    def __str__(self) -> str:
        self.validate()
        ids = np.array(self.sclst_ids)
        xs = np.array(self.x)
        return f"TPointList(gid={self.gid}, size={len(self)}):\n  ids: {ids}\n  x:   {xs}"
