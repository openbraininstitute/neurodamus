"""Main module for handling and instantiating synaptical connections and gap-junctions"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from itertools import chain
from os import path as ospath
from typing import TYPE_CHECKING

import numpy as np

from .connection import Connection, ReplayMode
from .core import MPI, NeuronWrapper as Nd, ProgressBarRank0 as ProgressBar, run_only_rank0
from .core.configuration import ConfigurationError, GlobalConfig, SimConfig
from .io.sonata_config import ConnectionTypes
from .io.synapse_reader import SonataReader
from .target_manager import TargetManager, TargetSpec
from .utils import compat
from .utils.logging import VERBOSE_LOGLEVEL, log_all, log_verbose
from .utils.pyutils import bin_search, dict_filter_map, gen_ranges
from .utils.timeit import timeit

if TYPE_CHECKING:
    from .utils.memory import DryRunStats


class ConnectionSet:
    """A dataset of connections.
    Several populations may exist with different seeds
    """

    def __init__(self, src_id, dst_id, conn_factory=Connection):
        # Connections indexed by post-gid, then ordered by pre-gid
        self.src_pop_id = src_id
        self.dst_pop_id = dst_id
        self.src_pop_name = None
        self.dst_pop_name = None
        self.virtual_source = False
        self._conn_factory = conn_factory
        self._connections_map = defaultdict(list)
        self._conn_count = 0

    def __getitem__(self, item):
        return self._connections_map[item]

    def items(self):
        """Iterate over the population as tuples (dst_gid, [connections])"""
        return self._connections_map.items()

    def target_gids(self):
        """Get the list of all targets gids in this Population"""
        return self._connections_map.keys()

    def all_connections(self):
        """Get an iterator over all the connections."""
        return chain.from_iterable(self._connections_map.values())

    def _find_connection(self, sgid, tgid, exact=True):
        """Finds a connection, given its source and destination gids.

        Returns:
            tuple: connection list and index.
                If the element doesnt exist, index depends on 'exact':
                None if exact=True, otherwise the insertion index.
        """
        cell_conns = self._connections_map[tgid]
        pos = 0
        if cell_conns:
            pos = bin_search(cell_conns, sgid, lambda x: x.sgid)
        if exact and (pos == len(cell_conns) or cell_conns[pos].sgid != sgid):
            # Not found
            return cell_conns, None
        return cell_conns, pos

    def get_connection(self, sgid, tgid):
        """Retrieves a connection from the pre and post gids.

        Returns:
            Connection: A connection object if it exists. None otherwise
        """
        conn_lst, idx = self._find_connection(sgid, tgid)
        return None if idx is None else conn_lst[idx]

    def store_connection(self, conn):
        """When we have created a new connection (sgid->tgid), store it
        in order in our structure for faster retrieval later

        Args:
            conn: The connection object to be stored
        """
        cell_conns, pos = self._find_connection(conn.sgid, conn.tgid, exact=False)
        if cell_conns and pos < len(cell_conns) and cell_conns[pos].sgid == conn.sgid:
            logging.error("Attempt to store existing connection: %d->%d", conn.sgid, conn.tgid)
            return
        self._conn_count += 1
        cell_conns.insert(pos, conn)

    def get_or_create_connection(self, sgid, tgid, **kwargs):
        """Returns a connection by pre-post gid, creating if required."""
        conns = self._connections_map[tgid]
        pos = 0
        if conns:
            # optimize for ordered insertion, and handle when sgid is not used
            last_conn = conns[-1]
            if last_conn.sgid in {sgid, None}:
                return last_conn
            if last_conn.sgid < sgid:
                pos = len(conns)
            else:
                pos = bin_search(conns, sgid, lambda x: x.sgid)
                if conns[pos].sgid == sgid:
                    return conns[pos]
        # Not found. Create & insert
        cur_conn = self._conn_factory(sgid, tgid, self.src_pop_id, self.dst_pop_id, **kwargs)
        conns.insert(pos, cur_conn)
        self._conn_count += 1
        return cur_conn

    def get_connections(self, post_gids, pre_gids=None):
        """Get all connections between groups of gids."""
        if isinstance(post_gids, int):
            if pre_gids is None:
                return self._connections_map[post_gids]
            if isinstance(pre_gids, int):
                elem = self.get_connection(pre_gids, post_gids)
                return (elem,) if elem is not None else ()

        post_gid_conn_lists = (
            self._connections_map.values()
            if post_gids is None
            else (self._connections_map[post_gids],)
            if isinstance(post_gids, int)
            else (self._connections_map[tgid] for tgid in post_gids)
        )
        if pre_gids is None:
            return chain.from_iterable(post_gid_conn_lists)
        if isinstance(pre_gids, int):
            # Return a generator which is employing bin search
            return (
                conns[posi]
                for conns in post_gid_conn_lists
                for posi in (bin_search(conns, pre_gids, lambda x: x.sgid),)
                if posi < len(conns) and conns[posi].sgid == pre_gids
            )
        # Generic case. Looks through all conns in selected tgids
        pre_gids = set(pre_gids)
        return (c for conns in post_gid_conn_lists for c in conns if c.sgid in pre_gids)

    def count(self):
        return self._conn_count

    def ids_match(self, population_ids, dst_second=None):
        """Whereas a given population_id selector matches population"""
        if isinstance(population_ids, tuple):
            expr_src, expr_dst = population_ids
        else:
            expr_src, expr_dst = (population_ids, dst_second)
        return (expr_src is None or expr_src == self.src_pop_id) and (
            expr_dst is None or expr_dst == self.dst_pop_id
        )

    def __str__(self):
        return (
            f"<ConnectionSet: {self.src_pop_id}-{self.dst_pop_id} "
            f"({self.src_pop_name}->{self.dst_pop_name})>"
        )

    def __repr__(self):
        return str(self)


class ConnectionManagerBase:
    """An abstract base class common to Synapse and GapJunction connections

    Connection Managers hold and manage connectivity among cell populations.
    For every src-dst pop pairs a new ConnectionManager is created.

    The only case it holds several ConnectionSets is for old-style projections (no population names)
    and possibily future support for multiple edge groups (e.g. from multiple files) so that
    additional connectivity for a pathway can be loaded and configured independently.
    NOTE: self._populations would require a new key format, not (src_pop, dst_pop)
    """

    CONNECTIONS_TYPE = None
    """The type of connections subclasses handle"""

    # Set depending Classes, customizable
    ConnectionSet = ConnectionSet
    SynapseReader = SonataReader
    conn_factory = Connection

    cell_manager = property(lambda self: self._cell_manager)
    src_cell_manager = property(lambda self: self._src_cell_manager)
    is_file_open = property(lambda self: bool(self._synapse_reader))
    src_pop_offset = property(lambda self: self._src_cell_manager.local_nodes.offset)
    target_pop_offset = property(lambda self: self._cell_manager.local_nodes.offset)
    src_node_population = property(lambda self: self._src_cell_manager.population_name)
    dst_node_population = property(lambda self: self._cell_manager.population_name)

    def __init__(self, circuit_conf, target_manager, cell_manager, src_cell_manager=None, **kw):
        """Base class c-tor for connections (Synapses & Gap-Junctions)

        Args:
            circuit_conf: A circuit config object where to get the synapse source from
                or None if the ConnecionManager is to be constructed empty
            target_manager: A target manager, where to query target cells
            cell_manager: Manager to query the local target cells
            src_cell_manager: Manager to query the local source cells [default: None -> internal]
            load_offsets: Whether to load synapse offsets

        """
        self._target_manager: TargetManager = target_manager
        self._cell_manager = cell_manager
        self._src_cell_manager = src_cell_manager or cell_manager

        self._populations = {}  # Multiple edge populations support. key is a tuple (src, dst)
        self._cur_population = None

        self._synapse_reader = None
        self._raw_gids = cell_manager.local_nodes.raw_gids()
        self._total_connections = 0
        self.circuit_conf = circuit_conf
        self._src_target_filter = None  # filter by src target in all_connect (E.g: GapJ)
        # Load offsets might be required globally. Conn managers shall honor this if possible
        self._load_offsets = kw.get("load_offsets", False)
        # An internal var to enable collection of synapse statistics to a Counter
        self._dry_run_stats: DryRunStats = kw.get("dry_run_stats")
        # For each tgid track "connected" sgids (dry-run)
        self._dry_run_conns = defaultdict(set)

    def __str__(self):
        name = self.__class__.__name__
        return f"<{name:s} | {self._src_cell_manager!s:s} -> {self._cell_manager!s:s}>"

    def open_edge_location(self, syn_source, _circuit_conf, **kw):
        edge_file, *pop = syn_source.split(":")
        pop_name = pop[0] if pop else None
        return self.open_synapse_file(edge_file, pop_name, **kw)

    def open_synapse_file(self, synapse_file, edge_population, *, src_pop_name=None, **_kw):
        """Initializes a reader for Synapses config objects and associated population

        Args:
            synapse_file: The nrn/edge file. For old nrn files it may be a dir.
            edge_population: The population of the edges
            src_name: The source pop name, normally matching that of the source cell manager
        """
        if not ospath.exists(synapse_file):
            raise ConfigurationError(f"Connectivity (Edge) file not found: {synapse_file}")
        if ospath.isdir(synapse_file):
            raise ConfigurationError("Edges source is a directory")

        self._synapse_reader = self._open_synapse_file(synapse_file, edge_population)
        if self._load_offsets and not self._synapse_reader.has_property("synapse_index"):
            raise Exception(
                "Synapse offsets required but not available. "
                "Please use a more recent version of neurodamus-core/synapse-tool"
            )

        self._init_conn_population(src_pop_name)
        self._unlock_all_connections()  # Allow appending synapses from new sources
        return synapse_file

    def _open_synapse_file(self, synapse_file, pop_name):
        logging.debug("Opening Synapse file %s, population: %s", synapse_file, pop_name)
        return self.SynapseReader(
            synapse_file, pop_name, extracellular_calcium=SimConfig.extracellular_calcium
        )

    def _init_conn_population(self, src_pop_name):
        if not src_pop_name:
            src_pop_name = self.src_node_population
        dst_pop_name = self.dst_node_population
        src_pop_id, dst_pop_id = self._compute_pop_ids(src_pop_name, dst_pop_name)

        if self._cur_population and src_pop_id == 0 and not src_pop_name:
            logging.warning(
                "Neither Sonata population nor populationID set. "
                "Edges will be merged with base circuit"
            )

        cur_pop = self.select_connection_set(src_pop_id, dst_pop_id)  # type: ConnectionSet
        cur_pop.src_pop_name = src_pop_name
        cur_pop.dst_pop_name = dst_pop_name
        cur_pop.virtual_source = (
            self._src_cell_manager.is_virtual or src_pop_name != self.src_node_population
        )
        logging.info("Loading connections to population: %s", cur_pop)

    @staticmethod
    def _compute_pop_ids(src_pop_name, dst_pop_name):
        """Compute pop id automatically base on population name."""

        def make_id(node_pop_name):
            pop_hash = hashlib.md5(node_pop_name.encode()).digest()
            return ((pop_hash[1] & 0x0F) << 8) + pop_hash[0]  # id: 12bit hash

        return make_id(src_pop_name), make_id(dst_pop_name)

    def select_connection_set(self, src_pop_id, dst_pop_id):
        """Select the active population of connections given src and dst node pop ids.
        `connect_all()` and `connect_group()` will apply only to the active population.

        Returns: The selected ConnectionSet, eventually created
        """
        self._cur_population = self.get_population(src_pop_id, dst_pop_id)
        return self._cur_population

    def get_population(self, src_pop_id, dst_pop_id):
        """Retrieves a connection set given node src and dst pop ids"""
        pop = self._populations.get((src_pop_id, dst_pop_id))
        if not pop:
            pop = self.ConnectionSet(src_pop_id, dst_pop_id, conn_factory=self.conn_factory)
            self._populations[src_pop_id, dst_pop_id] = pop
        return pop

    # NOTE: Several methods use a selector of the connectivity populations
    # which, to be backwards compat, can be a single ID of the src_population
    # or a tuple to specify source and destination

    def find_populations(self, population_ids):
        """Finds the populations that match a given population selector.

        Args:
            population_ids: A population ids selector. Accepted formats:

                - None: All
                - int: selects matching source population id.
                - tuple(src: Any, dst: Any): Specify source and dest.
                  Each value can also be None, e.g.: (None, 1) selects all
                  populations having post id 1
        """
        if (
            isinstance(population_ids, tuple)
            and population_ids[0] is not None
            and population_ids[1] is not None
        ):
            return [self._populations[population_ids]]
        return [pop for pop in self._populations.values() if pop.ids_match(population_ids)]

    def all_connections(self):
        """Retrieves all the existing connections"""
        return chain.from_iterable(pop.all_connections() for pop in self._populations.values())

    @property
    def connection_count(self):
        return self._total_connections

    @property
    def current_population(self):
        return self._cur_population

    def get_connections(self, post_gids, pre_gids=None, population_ids=None):
        """Retrieves all connections that match post and pre gids eventually
        in a subset of the populations.

        Note: Due to multi-population, a connection may not be unique
        for a given pre-post gid pair. As such get_connection() doesn't
        make sense anymore and this method shall be used instead.

        Args:
            post_gids: The target gids of the connections to search (None = All)
            pre_gids: idem for pre-gids. [Default: all)
            population_ids: A int/tuple of populations ids. Default: all

        """
        for pop in self.find_populations(population_ids):
            yield from pop.get_connections(post_gids, pre_gids)

    def create_connections(self, src_target=None, dst_target=None):
        """Creates connections according to loaded parameters in 'Connection'
        blocks of the config in the currently active ConnectionSet.

        If no Connection block relates to the current population, then load all
        edges. If a single blocks exists with Weight=0, skip creation entirely.

        NOTE: All connections respecting the src_target are retrieved
        and created, even if they use src cells which are NOT instantiated.
        This is to support replay and other stimulus which dont need the src cell.
        If only a subset of connections is wanted, they can be filtered by specifying
        the "source" attribute of the respective connection blocks.

        Args:
            src_target: Target name to restrict creating connections coming from it
            dst_target: Target name to restrict creating connections going into it
        """
        conn_src_spec = TargetSpec(src_target)  # instantiate all from src
        conn_src_spec.population = self.current_population.src_pop_name
        conn_dst_spec = TargetSpec(dst_target or self.cell_manager.circuit_target)
        conn_dst_spec.population = self.current_population.dst_pop_name
        this_pathway = {"Source": str(conn_src_spec), "Destination": str(conn_dst_spec)}
        matching_conns = [
            conn
            for conn in SimConfig.connections.values()
            if self._target_manager.pathways_overlap(conn, this_pathway)
        ]
        if not matching_conns:
            logging.info("No matching Connection blocks. Loading all synapses...")
            self.connect_all()
            return

        # if we have a single connect block with weight=0, skip synapse creation entirely
        if len(matching_conns) == 1 and matching_conns[0].get("Weight") == 0.0:
            logging.warning("SKIPPING Connection create since they have invariably weight=0")
            return

        logging.info("Creating group connections (%d groups match)", len(matching_conns))
        for conn_conf in matching_conns:
            if "Delay" in conn_conf and conn_conf["Delay"] > 0:
                # Delayed connections are for configuration only, not creation
                continue

            # check if we are not supposed to create (only configure later)
            if conn_conf.get("CreateMode") == "NoCreate":
                continue

            conn_src = conn_conf["Source"]
            conn_dst = conn_conf["Destination"]
            synapse_id = conn_conf.get("SynapseID")
            mod_override = conn_conf.get("ModOverride")
            self.connect_group(conn_src, conn_dst, synapse_id, mod_override)

    def configure_connections(self, conn_conf):
        """Configure-only circuit connections according to a config Connection block

        Args:
            conn_conf: The configuration block (dict)
        """
        log_msg = " * Pathway {:s} -> {:s}".format(conn_conf["Source"], conn_conf["Destination"])

        if "Delay" in conn_conf and conn_conf["Delay"] > 0:
            log_msg += f":\t[DELAYED] t={conn_conf['Delay']:g}, weight={conn_conf['Weight']:g}"
            configured_conns = self.setup_delayed_connection(conn_conf)
        else:
            if "SynapseConfigure" in conn_conf:
                log_msg += ":\tconfigure with '{:s}'".format(conn_conf["SynapseConfigure"])
            if "NeuromodStrength" in conn_conf:
                log_msg += "\toverwrite NeuromodStrength = {:g}".format(
                    conn_conf["NeuromodStrength"]
                )
            if "NeuromodDtc" in conn_conf:
                log_msg += "\toverwrite NeuromodDtc = {:g}".format(conn_conf["NeuromodDtc"])
            configured_conns = self.configure_group(conn_conf)

        all_ranks_total = MPI.allreduce(configured_conns, MPI.SUM)
        if all_ranks_total > 0:
            logging.info(log_msg)
            logging.info(" => Configured %s connections", all_ranks_total)

    def setup_delayed_connection(self, conn_config):
        raise NotImplementedError(
            f"Manager {self.__class__.__name__} doesn't implement delayed connections"
        )

    def connect_all(self, weight_factor=1):
        """For every gid access its synapse parameters and instantiate all synapses.

        Args:
            weight_factor: Factor to scale all netcon weights (default: 1)
            only_gids: Create connections only for these tgids (default: Off)
        """
        if SimConfig.dry_run:
            syn_count = self._get_conn_stats(None)
            log_all(VERBOSE_LOGLEVEL, "[Rank %d] Synapse count: %d", MPI.rank, syn_count)
            self._dry_run_stats.synapse_counts[self.CONNECTIONS_TYPE] += syn_count
            return

        conn_options = {"weight_factor": weight_factor}
        pop = self._cur_population

        for sgid, tgid, syns_params, extra_params, offset in self._iterate_conn_params(
            src_target=self._src_target_filter, dst_target=None, show_progress=True
        ):
            if self._load_offsets:
                conn_options["synapses_offset"] = extra_params["synapse_index"][0]
            # Create all synapses. No need to lock since the whole file is consumed
            cur_conn = pop.get_or_create_connection(sgid, tgid, **conn_options)
            self._add_synapses(cur_conn, syns_params, None, offset)

    def connect_group(
        self, conn_source, conn_destination, synapse_type_restrict=None, mod_override=None
    ):
        """Instantiates pathway connections & synapses given src-dst

        Args:
            conn_source (str): The target name of the source cells
            conn_destination (str): The target of the destination cells
            synapse_type_restrict(int): Create only given synType synapses
            mod_override (str): ModOverride given for this connection group
        """
        conn_kwargs = {}
        conn_pop = self._cur_population
        dst_pop_name = self.dst_node_population
        src_pop_name = self.src_node_population
        logging.debug("Connecting group %s -> %s", conn_source, conn_destination)
        src_tspec = TargetSpec(conn_source)
        dst_tspec = TargetSpec(conn_destination)
        src_target = src_tspec.name and self._target_manager.get_target(src_tspec, src_pop_name)
        dst_target = dst_tspec.name and self._target_manager.get_target(dst_tspec, dst_pop_name)

        if (src_target and src_target.is_void()) or (dst_target and dst_target.is_void()):
            logging.debug(
                "Skip void connectivity for current connectivity: %s - %s",
                conn_source,
                conn_destination,
            )
            return

        if SimConfig.dry_run:
            syn_count = self._get_conn_stats(dst_target, src_target)
            log_all(VERBOSE_LOGLEVEL, "%s-> %s: %d", conn_pop.src_name, conn_destination, syn_count)
            self._dry_run_stats.synapse_counts[self.CONNECTIONS_TYPE] += syn_count
            return

        for sgid, tgid, syns_params, extra_params, offset in self._iterate_conn_params(
            src_target, dst_target, show_progress=None, mod_override=mod_override
        ):
            if sgid == tgid:
                logging.warning("Making connection within same Gid: %d", sgid)
            if self._load_offsets:
                conn_kwargs["synapses_offset"] = extra_params["synapse_index"][0]

            cur_conn = conn_pop.get_or_create_connection(sgid, tgid, **conn_kwargs)
            if cur_conn.locked:
                continue
            self._add_synapses(cur_conn, syns_params, synapse_type_restrict, offset)
            cur_conn.locked = True

    def _add_synapses(self, cur_conn: Connection, syns_params, syn_type_restrict=None, base_id=0):
        if syn_type_restrict:
            syns_params = syns_params[syns_params["synType"] != syn_type_restrict]
        if len(syns_params) == 0:
            return
        if SimConfig.crash_test_mode:
            cur_conn.add_single(self._cell_manager, syns_params[0], base_id)
        else:
            cur_conn.add_synapses(self._target_manager, syns_params, base_id)

    class ConnDebugger:
        __slots__ = ["yielded_src_gids"]

        def __init__(self):
            self.yielded_src_gids = compat.array("i") if GlobalConfig.debug_conn else None

        def register(self, sgid, base_tgid, syns_params):
            if not (debug_conn := GlobalConfig.debug_conn):
                return
            if debug_conn == [base_tgid]:
                self.yielded_src_gids.append(sgid)
            elif debug_conn == [sgid, base_tgid]:
                log_all(
                    logging.DEBUG, "Connection (%d-%d). Params:\n%s", sgid, base_tgid, syns_params
                )

        def __del__(self):
            if self.yielded_src_gids:
                log_all(logging.DEBUG, "Source GIDs for debug cell: %s", self.yielded_src_gids)

    @staticmethod
    def _get_allowed_ranges(src_target, sgids, sgids_ranges, conn_count):
        """Return n_yielded_conns and allowed_ranges, handling src_target=None.

        Helper function for _iterate_conn_params
        """
        if src_target:
            unique_sgids = sgids[sgids_ranges[:-1]]
            allowed_sgids = set(unique_sgids[src_target.contains(unique_sgids, raw_gids=True)])
            allowed_ranges = [
                (sgids_ranges[i], sgids_ranges[i + 1])
                for i in range(conn_count)
                if sgids[sgids_ranges[i]] in allowed_sgids
            ]
            n_yielded_conns = len(allowed_sgids)
        else:
            n_yielded_conns = conn_count
            allowed_ranges = [(sgids_ranges[i], sgids_ranges[i + 1]) for i in range(conn_count)]

        return n_yielded_conns, allowed_ranges

    @staticmethod
    def _compute_sgids_ranges(syns_params):
        """Compute source GIDs, their change points, and total connections count.

        Helper function for _iterate_conn_params
        """
        sgids = syns_params[syns_params.dtype.names[0]].astype("int64")
        sgids_ranges = np.diff(sgids, prepend=np.nan, append=np.nan).nonzero()[0]
        conn_count = len(sgids_ranges) - 1
        return sgids, sgids_ranges, conn_count

    def _get_extra_fields(self, base_tgid):
        """Get extra fields for the synapse parameters, e.g. synapse_index.

        Helper function for _iterate_conn_params
        """
        if self._load_offsets:
            syn_index = self._synapse_reader.get_property(base_tgid, "synapse_index")
            return {"synapse_index": syn_index}
        return {}

    def _iterate_conn_params(  # noqa: PLR0914
        self,
        src_target,
        dst_target,
        show_progress=None,
        mod_override=None,
    ):
        """A generator which loads synapse data and yields tuples(sgid, tgid, synapses)

        Args:
            src_target: the target to filter the source cells, or None
            dst_target: the target to filter the destination cells, or None
            show_progress: Display a progress bar as tgids are processed
        """
        AUTO_PROGRESS_THRESHOLD = 50
        if (src_target and src_target.is_void()) or (dst_target and dst_target.is_void()):
            return

        gids = self._raw_gids

        if dst_target:
            gids = np.intersect1d(gids, dst_target.get_raw_gids())

        created_conns_0 = self._cur_population.count()
        sgid_offset = self.src_pop_offset
        tgid_offset = self.target_pop_offset

        self._synapse_reader.configure_override(mod_override)
        self._synapse_reader.preload_data(gids, minimal_mode=SimConfig.cli_options.crash_test)

        # NOTE: This routine is quite critical, sitting at the core of synapse processing
        # so it has been carefully optimized with numpy vectorized operations, even if
        # it might lose some readability.
        # For each tgid we obtain the synapse parameters as a record array. We then split it,
        # without copying, yielding ranges (views) of it.
        if show_progress is None:
            show_progress = len(gids) >= AUTO_PROGRESS_THRESHOLD

        gids = ProgressBar.iter(gids, name="Loading") if show_progress else gids

        for base_tgid in gids:
            tgid = base_tgid + tgid_offset
            syns_params = self._synapse_reader.get_synapse_parameters(base_tgid)
            logging.debug("GID %d Syn count: %d", tgid, len(syns_params))

            sgids, sgids_ranges, conn_count = self._compute_sgids_ranges(syns_params)
            conn_debugger = self.ConnDebugger()
            if conn_count == 0:
                logging.debug("No synapses for GID %d. Nothing to do.", tgid)
                continue

            extra_fields = self._get_extra_fields(base_tgid)

            # We yield ranges of contiguous parameters belonging to the same connection,
            # and given we have data for a single tgid, enough to group by sgid.
            # The first row of a range is found by numpy.diff

            n_yielded_conns, allowed_ranges = self._get_allowed_ranges(
                src_target, sgids, sgids_ranges, conn_count
            )

            for range_start, range_end in allowed_ranges:
                sgid = int(sgids[range_start])
                final_sgid = sgid + sgid_offset
                syn_params = syns_params[range_start:range_end]
                extra_params = (
                    extra_fields
                    and {  # reuse empty {}. Dont modify later!
                        name: prop[range_start:range_end] for name, prop in extra_fields.items()
                    }
                )
                conn_debugger.register(sgid, base_tgid, syn_params)
                yield final_sgid, tgid, syn_params, extra_params, range_start

            logging.debug(
                " > Yielded %d out of %d connections. (Filter by src Target: %s)",
                n_yielded_conns,
                conn_count,
                src_target and src_target.name,
            )

        created_conns = self._cur_population.count() - created_conns_0
        self._total_connections += created_conns

        all_created = MPI.allreduce(created_conns, MPI.SUM)
        if all_created:
            pathway_repr = "[ALL]"
            if src_target and dst_target:
                pathway_repr = f"Pathway {src_target.name} -> {dst_target.name}"
            logging.info(" * %s. Created %d connections", pathway_repr, all_created)

    def _get_conn_stats(self, dst_target, src_target=None):  # noqa: PLR0914
        """Estimates the number of synapses for the given destination and source nodesets

        Args:
            dst_nodeset: The target to estimate synapses for
            src_nodeset: The source nodes allowed for the given synapses

        Returns:
            The estimated number of synapses which would be created

        """
        BLOCK_BASE_SIZE = 5000
        SAMPLED_CELLS_PER_BLOCK = 100

        # Get the raw gids for the destination target (in this rank)
        local_gids = dst_target.get_local_gids(raw_gids=True) if dst_target else self._raw_gids
        if not len(local_gids):  # Target is empty in this rank
            logging.debug("Skipping group: no cells!")
            return 0

        total_estimate = 0
        gids_per_metype = self._dry_run_stats.pop_metype_gids[self.dst_node_population]

        # NOTE:
        #  - Estimation (and extrapolation) is performed per metype since properties can vary
        #  - Consider only the cells for the current target

        for metype, all_me_gids in gids_per_metype.items():
            me_gids = set(all_me_gids).intersection(local_gids)
            me_gids_count = len(me_gids)
            if not me_gids_count:
                logging.debug("Skipping metype '%s': no cells!", metype)
                continue

            logging.debug("Metype %s", metype)
            me_gids = np.fromiter(me_gids, dtype="uint32")

            # NOTE:
            # Process the first 100 cells from increasingly large blocks
            #  - Takes advantage of data locality
            #  - Blocks increase as a geometric progression for handling very large sets

            metype_estimate = 0

            for (
                start,
                stop,
            ) in gen_ranges(me_gids_count, BLOCK_BASE_SIZE, block_increase_rate=1.1):
                logging.debug(" - Processing range %d:%d", start, stop)
                block_len = stop - start
                sample = me_gids[start : (start + SAMPLED_CELLS_PER_BLOCK)]
                sample_len = len(sample)
                if not sample_len:
                    continue

                sample_counts = self._synapse_reader.get_conn_counts(sample)
                total_connections = 0
                selected_conn_count = 0
                new_conn_count = 0  # Let's count those which were not "created" before
                new_syns_count = 0

                for tgid, tgid_conn_counts in sample_counts.items():
                    total_connections += len(tgid_conn_counts)
                    if src_target:
                        conn_sgids = np.fromiter(tgid_conn_counts.keys(), dtype="uint32")
                        sgids_in_target = conn_sgids[src_target.contains(conn_sgids, raw_gids=True)]
                    else:
                        sgids_in_target = tgid_conn_counts.keys()

                    selected_conn_count += len(sgids_in_target)
                    tgid_connected_sgids = self._dry_run_conns[tgid]

                    for sgid in sgids_in_target:
                        if sgid not in tgid_connected_sgids:
                            new_conn_count += 1
                            new_syns_count += tgid_conn_counts[sgid]
                            tgid_connected_sgids.add(int(sgid))

                logging.debug(
                    " - Connections (new/selected/total): %d / %d / %d ",
                    new_conn_count,
                    selected_conn_count,
                    total_connections,
                )
                block_syns_per_cell = new_syns_count / sample_len
                logging.debug(" - Synapses: %d (Avg: %.2f)", new_syns_count, block_syns_per_cell)
                metype_estimate += block_syns_per_cell * block_len

            # Info on the whole metype (subject to selected target)
            # Due to the fact that the same metype might be target of several projections
            #   we have to sum the averages
            average_syns_per_cell = metype_estimate / me_gids_count
            self._dry_run_stats.metype_cell_syn_average[metype] += average_syns_per_cell
            log_all(
                logging.DEBUG,
                "%s: Average syns/cell: %.1f, Estimated total: %d ",
                metype,
                average_syns_per_cell,
                metype_estimate,
            )
            total_estimate += metype_estimate

        return int(total_estimate)

    def get_target_connections(
        self, src_target_name, dst_target_name, selected_gids=None, conn_population=None
    ):
        """Retrives the connections between src-dst cell targets

        Args:
             selected_gids: (optional) post gids to select (original, w/o offsetting)
             conn_population: restrict the set of connections to be returned
        """
        src_target_spec = TargetSpec(src_target_name)
        dst_target_spec = TargetSpec(dst_target_name)

        src_target = (
            self._target_manager.get_target(src_target_spec)
            if src_target_spec.name is not None
            else None
        )
        assert dst_target_spec.name, "No target specified for `get_target_connections`"
        dst_target = self._target_manager.get_target(dst_target_spec)
        if (src_target and src_target.is_void()) or dst_target.is_void():
            return

        tgid_offset = self.target_pop_offset
        conn_populations: list[ConnectionSet] = (
            (conn_population,) if conn_population is not None else self._populations.values()
        )

        # temporary set for faster lookup
        src_gids = src_target and set(src_target.get_gids())

        for population in conn_populations:
            logging.debug("Connections from population %s", population)
            tgids = np.fromiter(population.target_gids(), "uint32")
            tgids = np.intersect1d(tgids, dst_target.get_gids())
            if selected_gids:
                tgids = np.intersect1d(tgids, selected_gids + tgid_offset)
            for conn in population.get_connections(tgids):
                if src_target is None or conn.sgid in src_gids:
                    yield conn

    def configure_group(self, conn_config, gidvec=None):
        """Configure connections according to a config Connection block

        Args:
            conn_config: The connection configuration dict
            gidvec: A restricted set of gids to configure (original, w/o offsetting)
        """
        src_target = conn_config["Source"]
        dst_target = conn_config["Destination"]
        syn_params = dict_filter_map(
            conn_config,
            {
                "Weight": "weight_factor",
                "SpontMinis": "minis_spont_rate",
                "SynDelayOverride": "syndelay_override",
                "NeuromodStrength": "neuromod_strength",
                "NeuromodDtc": "neuromod_dtc",
            },
        )

        # Load eventual mod override helper
        if "ModOverride" in conn_config:
            logging.info("   => Overriding mod: %s", conn_config["ModOverride"])
            override_helper = conn_config["ModOverride"] + "Helper"
            Nd.load_hoc(override_helper)
            assert hasattr(Nd.h, override_helper), (
                "ModOverride helper doesn't define hoc template: " + override_helper
            )

        configured_conns = 0
        for conn in self.get_target_connections(src_target, dst_target, gidvec):
            for key, val in syn_params.items():
                setattr(conn, key, val)
            if "ModOverride" in conn_config:
                conn.mod_override = conn_config["ModOverride"]
            if "SynapseConfigure" in conn_config:
                conn.add_synapse_configuration(conn_config["SynapseConfigure"])
            configured_conns += 1
        return configured_conns

    def restart_events(self):
        """After restore, restart the artificial events (replay and spont minis)"""
        for conn in self.all_connections():
            conn.restart_events()

    def _unlock_all_connections(self):
        """Unlock all, mainly when we load a new connectivity source"""
        for conn in self.all_connections():
            conn.locked = False

    def finalize(self, base_seed=0, *, _conn_type="synapses", **conn_params):
        """Instantiates the netcons and Synapses for all connections.

        Note: All weight scalars should have their final values.

        Args:
            base_seed: optional argument to adjust synapse RNGs (default=0)
            _conn_type: (Internal) A string repr of the connectivity type
            conn_params: Additional finalize parameters for the specific _finalize_conns
                E.g. replay_mode (Default: Auto-Detect) Use DISABLED to skip replay
                and COMPLETE to instantiate VecStims in all synapses

        """
        self._synapse_reader = None  # Destroy to release memory (all cached params)
        logging.info("Instantiating %s... Params: %s", _conn_type, str(conn_params))
        n_created_conns = 0

        for popid, pop in self._populations.items():
            attach_src = pop.src_pop_id == 0 or not pop.virtual_source  # real populations
            conn_params["attach_src_cell"] = attach_src
            logging.info(
                " * Connections among %s -> %s, attach src: %s",
                pop.src_pop_name or "(base)",
                pop.dst_pop_name or "(base)",
                attach_src,
            )

            for tgid, conns in ProgressBar.iter(pop.items(), name="Pop:" + str(popid)):
                n_created_conns += self._finalize_conns(tgid, conns, base_seed, **conn_params)

        all_ranks_total = MPI.allreduce(n_created_conns, MPI.SUM)
        logging.info(" => Created %d %s", all_ranks_total, _conn_type)
        return all_ranks_total

    def _finalize_conns(self, tgid, conns, base_seed, *, reverse=False, **kwargs):
        """Low-level handling of finalizing connections belonging to a target gid.
        By default it calls finalize on each cell.
        """
        # Note: *kwargs normally contains 'replay_mode' but may differ for other types
        metype = self._cell_manager.get_cell(tgid)
        n_created_conns = 0
        if reverse:
            conns = reversed(conns)
        for conn in conns:  # type: Connection
            syn_count = conn.finalize(metype, base_seed, **kwargs)
            logging.debug("Instantiated conn %s: %d synapses", conn, syn_count)
            n_created_conns += syn_count
        return n_created_conns

    def replay(self, *_, **_kw):
        logging.warning("Replay is not available in %s", self.__class__.__name__)


# ##############
# Helper methods
# ##############


def edge_node_pop_names(edge_file, edge_pop_name, src_pop_name=None, dst_pop_name=None) -> tuple:
    """Find the node populations names.

    Args:
        edge_file: The edge file to extract the population names from
        edge_pop_name: The name of the edge population
    Returns: tuple of the src-dst population names.
    """
    src_dst_pop_names = _edge_meta_get_node_populations(edge_file, edge_pop_name)
    if src_dst_pop_names:
        if src_pop_name is None:
            src_pop_name = src_dst_pop_names[0]
        if dst_pop_name is None:
            dst_pop_name = src_dst_pop_names[1]
    return src_pop_name, dst_pop_name


@run_only_rank0
def _edge_meta_get_node_populations(edge_file, edge_pop_name) -> tuple:
    import libsonata

    edge_storage = libsonata.EdgeStorage(edge_file)
    if not edge_pop_name:
        assert len(edge_storage.population_names) == 1, (
            "multi-population edges require manual selection"
        )
        edge_pop_name = next(iter(edge_storage.population_names))

    edge_pop = edge_storage.open_population(edge_pop_name)
    return (edge_pop.source, edge_pop.target)


# ######################################################################
# SynapseRuleManager
# ######################################################################
class SynapseRuleManager(ConnectionManagerBase):
    """The SynapseRuleManager is designed to encapsulate the creation of
    synapses for BlueBrain simulations, handling the data coming from
    the circuit file. If the config file provides any Connection
    Rules, those override which synapses are created.

    Note that the Connection rules are processed with the assumption
    that they come in the config file from more general to more specific.
    E.g.: A column->column connection should come before
    layer 4 -> layer 2 which should come before L4PC -> L2PC.

    Once all synapses are prepared with final weights, the Netcons can be
    created.
    """

    CONNECTIONS_TYPE = ConnectionTypes.Synaptic

    def __init__(self, circuit_conf, target_manager, cell_manager, src_cell_manager=None, **kw):
        """Initializes a Connection/Edge manager for standard METype synapses

        Args:
            circuit_conf: The configuration object/dict
            target_manager: The (hoc-level) target manager
            cell_manager: The destination cell population manager
            src_cell_manager: The source cell population manager [Default: same as destination]
        """
        super().__init__(circuit_conf, target_manager, cell_manager, src_cell_manager, **kw)
        # SynapseRuleManager opens synapse file and init populations
        syn_source = circuit_conf.get("nrnPath")
        if syn_source:
            logging.info("Init %s. Options: %s", type(self).__name__, kw)
            self.open_edge_location(syn_source, circuit_conf, **kw)

    def finalize(self, base_seed=0, **kwargs):
        """Create the actual synapses and netcons. See super() docstring"""
        kwargs.setdefault("replay_mode", ReplayMode.AS_REQUIRED)
        super().finalize(base_seed, **kwargs)

    def _finalize_conns(self, tgid, conns, base_seed, **kw):
        # Note: (Compat) neurodamus hoc finalizes connections in reversed order.
        return super()._finalize_conns(tgid, conns, base_seed, reverse=True, **kw)

    def setup_delayed_connection(self, conn_config):
        """Setup delayed connection weights for synapse initialization.

        Find source and target gids and the associated connection,
        and add the delay and weight to their delay vectors.

        Args:
            conn_config: Connection configuration parsed from sonata config
        """
        src_target_name = conn_config["Source"]
        dst_target_name = conn_config["Destination"]
        delay = conn_config["Delay"]
        new_weight = conn_config.get("Weight", 0.0)

        configured_conns = 0
        for conn in self.get_target_connections(src_target_name, dst_target_name):
            conn.add_delayed_weight(delay, new_weight)
            configured_conns += 1
        return configured_conns

    @timeit(name="Replay inject")
    def replay(self, spike_manager, src_target_name, dst_target_name, start_delay=0.0):
        """Create special netcons to trigger timed spikes on those synapses.

        Args:
            spike_manager: map of gids (pre-synaptic) with spike times
            src_target_name: Source population:target of the replay connections
            dst_target_name: Target whose gids should be replayed
            start_delay: Dont deliver events before t=start_delay
        """
        log_verbose("Applying replay map with %d src cells...", len(spike_manager))
        replayed_count = 0

        # Dont deliver events in the past
        if Nd.t > start_delay:
            start_delay = Nd.t
            log_verbose("Restore: Delivering events only after t=%.4f", start_delay)

        src_pop_offset = self.src_pop_offset

        for conn in self.get_target_connections(src_target_name, dst_target_name):
            raw_sgid = conn.sgid - src_pop_offset
            if raw_sgid not in spike_manager:
                continue
            conn.replay(spike_manager[raw_sgid], start_delay)
            replayed_count += 1

        total_replays = MPI.allreduce(replayed_count, MPI.SUM)
        if MPI.rank == 0:
            if total_replays == 0:
                logging.warning("No connections were injected replay stimulus")
            else:
                logging.info(" => Replaying on %d connections", total_replays)
        return total_replays
