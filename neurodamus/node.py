# Neurodamus
# Copyright 2018 - Blue Brain Project, EPFL
from __future__ import annotations

import gc
import glob
import itertools
import logging
import math
import os
import shutil
import typing
from collections import defaultdict
from contextlib import contextmanager
from os import path as ospath
from pathlib import Path

import libsonata

# Internal Plugins
from . import ngv as _ngv
from .cell_distributor import (
    CellDistributor,
    GlobalCellManager,
    LoadBalance,
    LoadBalanceMode,
    VirtualCellPopulation,
)
from .connection_manager import SynapseRuleManager, edge_node_pop_names
from .core import (
    MPI,
    NeuronWrapper as Nd,
    SimulationProgress,
    mpi_no_errors,
    return_neuron_timings,
    run_only_rank0,
)
from .core._engine import EngineBase
from .core._shmutils import SHMUtil
from .core.configuration import (
    CircuitConfig,
    ConfigurationError,
    Feature,
    GlobalConfig,
    SimConfig,
    _SimConfig,
    get_debug_cell_gids,
    make_circuit_config,
)
from .core.coreneuron_configuration import CompartmentMapping, CoreConfig
from .core.nodeset import PopulationNodes
from .gap_junction import GapJunctionManager
from .io.sonata_config import ConnectionTypes
from .modification_manager import ModificationManager
from .neuromodulation_manager import NeuroModulationManager
from .replay import MissingSpikesPopulationError, SpikeManager
from .report import create_report
from .stimulus_manager import StimulusManager
from .target_manager import TargetManager, TargetSpec
from .utils.logging import log_stage, log_verbose
from .utils.memory import DryRunStats, free_event_queues, pool_shrink, print_mem_usage, trim_memory
from .utils.timeit import TimerManager, timeit
from neurodamus.utils.pyutils import rmtree


class METypeEngine(EngineBase):
    CellManagerCls = CellDistributor
    InnerConnectivityCls = SynapseRuleManager
    ConnectionTypes = {
        None: SynapseRuleManager,
        ConnectionTypes.Synaptic: SynapseRuleManager,
        ConnectionTypes.GapJunction: GapJunctionManager,
        ConnectionTypes.NeuroModulation: NeuroModulationManager,
    }
    CircuitPrecedence = 0


class CircuitManager:
    """Holds and manages populations and associated nodes and edges

    All nodes must have a name or read from sonata pop name
    As so, Sonata is preferred when using multiple node files
    """

    def __init__(self):
        self.node_managers = {}  # dict {pop_name -> cell_manager}  # nrn pop is None
        self.virtual_node_managers = {}  # same, but for virtual ones (no cells)
        # dict {(src_pop, dst_pop) -> list[synapse_manager]}
        self.edge_managers = defaultdict(list)
        self.alias = {}  # dict {name -> pop_name}
        self.global_manager = GlobalCellManager()
        self.global_target = TargetManager.create_global_target()

    def initialized(self):
        return bool(self.node_managers)

    def register_node_manager(self, cell_manager):
        pop = cell_manager.population_name
        if pop in self.node_managers:
            raise ConfigurationError(f"Already existing node manager for population {pop}")
        self.node_managers[pop] = cell_manager
        self.alias[cell_manager.circuit_name] = pop
        self.global_manager.register_manager(cell_manager)
        if cell_manager.is_initialized():
            self.global_target.append_nodeset(cell_manager.local_nodes)

    def _new_virtual_node_manager(self, circuit):
        """Instantiate a new virtual node manager explicitly."""
        storage = libsonata.NodeStorage(circuit.CellLibraryFile)
        pop_name, _ = circuit.CircuitTarget.split(":")  # Sonata config fills population
        node_size = storage.open_population(pop_name).size
        gid_vec = list(range(1, node_size + 1))
        virtual_cell_manager = VirtualCellPopulation(pop_name, gid_vec)
        self.virtual_node_managers[pop_name] = virtual_cell_manager
        self.global_target.append_nodeset(virtual_cell_manager.local_nodes)
        return virtual_cell_manager

    @staticmethod
    def new_node_manager_bare(circuit: CircuitConfig, target_manager, run_conf, **kwargs):
        engine = circuit.Engine or METypeEngine
        CellManagerCls = engine.CellManagerCls or CellDistributor
        return CellManagerCls(circuit, target_manager, run_conf, **kwargs)

    def new_node_manager(self, circuit, target_manager, run_conf, *, load_balancer=None, **kwargs):
        if circuit.get("PopulationType") == "virtual":
            return self._new_virtual_node_manager(circuit)
        cell_manager = self.new_node_manager_bare(circuit, target_manager, run_conf, **kwargs)
        cell_manager.load_nodes(load_balancer, **kwargs)
        self.register_node_manager(cell_manager)
        return cell_manager

    def get_node_manager(self, name):
        name = self.alias.get(name, name)
        return self.node_managers.get(name)

    def has_population(self, pop_name):
        return pop_name in self.node_managers

    def unalias_pop_keys(self, source, destination):
        """Un-alias population names"""
        return self.alias.get(source, source), self.alias.get(destination, destination)

    def get_edge_managers(self, source, destination):
        edge_pop_keys = self.unalias_pop_keys(source, destination)
        return self.edge_managers.get(edge_pop_keys) or []

    def get_edge_manager(self, source, destination, conn_type=SynapseRuleManager):
        managers = [
            manager
            for manager in self.get_edge_managers(source, destination)
            if isinstance(manager, conn_type)
        ]
        return managers[0] if managers else None

    def get_create_edge_manager(
        self, conn_type, source, destination, src_target, manager_args=(), **kw
    ):
        source, destination = self.unalias_pop_keys(source, destination)
        manager = self.get_edge_manager(source, destination, conn_type)
        if manager:
            return manager

        if not self.has_population(destination):
            raise ConfigurationError("Can't find projection Node population: " + destination)

        src_manager = self.node_managers.get(source) or self.virtual_node_managers.get(source)
        if src_manager is None:  # src manager may not exist -> virtual
            log_verbose("No known population %s. Creating Virtual src for projection", source)
            if conn_type not in {SynapseRuleManager, _ngv.GlioVascularManager}:
                raise ConfigurationError("Custom connections require instantiated source nodes")
            src_manager = VirtualCellPopulation(source, None, src_target.name)

        target_cell_manager = kw["cell_manager"] = self.node_managers[destination]
        kw["src_cell_manager"] = src_manager
        manager = conn_type(*manager_args, **kw)
        self.edge_managers[source, destination].append(manager)
        target_cell_manager.register_connection_manager(manager)
        return manager

    def all_node_managers(self):
        return self.node_managers.values()

    def all_synapse_managers(self):
        return itertools.chain.from_iterable(self.edge_managers.values())

    @staticmethod
    @run_only_rank0
    def write_population_offsets(pop_offsets, alias_pop, virtual_pop_offsets):
        """Write population_offsets where appropriate

        It is needed for retrieving population offsets for reporting and replay at restore time.

        Format population name::gid offset::population alias
        The virtual population offset is also written for synapse replay in restore.
        The data comes from outside because pop_offsets are not initialized
        in a restore scenario.
        """
        # populations_offset is necessary in output_path
        output_path = SimConfig.populations_offset_output_path(create=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.writelines(
                "{}::{}::{}\n".format(pop or " ", pop_offsets[pop], alias or " ")
                for alias, pop in alias_pop.items()
            )
            f.writelines(
                "{}::{}::{}\n".format(pop, offset, "virtual")
                for pop, offset in virtual_pop_offsets.items()
            )

        # Add a file in save_path too if required
        if SimConfig.save:
            save_path = SimConfig.populations_offset_save_path(create=True)
            shutil.copy(output_path, save_path)

    def get_population_offsets(self):
        pop_offsets = {
            pop_name: node_manager.local_nodes.offset
            for pop_name, node_manager in self.node_managers.items()
        }
        alias_pop = dict(self.alias)
        return pop_offsets, alias_pop

    def get_virtual_population_offsets(self):
        pop_offsets = {
            pop_name: node_manager.local_nodes.offset
            for pop_name, node_manager in self.virtual_node_managers.items()
        }
        return pop_offsets

    @classmethod
    def read_population_offsets(cls, file_path=None):
        """Read population offsets from populations_offset.dat"""
        pop_offsets = {}
        alias_pop = {}
        virtual_pop_offsets = {}
        with open(file_path or SimConfig.populations_offset_restore_path(), encoding="utf-8") as f:
            for line in f:
                pop, offset, alias = line.strip().split("::")
                pop = pop or None
                alias = alias or None
                if alias == "virtual":
                    virtual_pop_offsets[pop] = int(offset)
                else:
                    pop_offsets[pop] = int(offset)
                    alias_pop[alias] = pop

        return pop_offsets, alias_pop, virtual_pop_offsets

    def __del__(self):
        """De-init. Edge managers must be destructed first"""
        del self.edge_managers
        del self.virtual_node_managers
        del self.node_managers


class ReportsCumulativeError(Exception):
    def __init__(self, errors):
        self.errors = errors
        super().__init__(self._format_message())

    def _format_message(self):
        messages = []
        for func_name, value, err in self.errors:
            messages.append(f"{func_name}({value}): {type(err).__name__} -> {err}")
        return "enable_reports failed with multiple errors:\n" + "\n".join(messages)


class ReportSetupError(Exception):
    pass


class Node:
    """The Node class is the main entity for a distributed Neurodamus execution.

    Note that this concept of a "Node" differs from both an MPI node, which
    refers to a process in a parallel computing environment, and a node in the
    circuit graph, which represents an individual element or component within
    the simulation's neural network.

    It serves as the orchestrator for the entire simulation, managing the
    parallel execution of the model and distributing the cells across different
    computational ranks. As the primary control structure, Node is responsible
    for coordinating various components involved in the simulation.

    Internally, the Node class instantiates and manages parallel structures,
    dividing the simulation workload among multiple ranks. With the introduction
    of the concept of multiple populations (also known as multi-circuit), the
    Node class takes partial responsibility for handling this logic, aided by
    the :class:`neurodamus.node.CircuitManager` class (accessible via the
    `circuits` property), which manages the different node and edge managers.

    While many lower-level details of the Node's functionality are encapsulated
    within dedicated helper classes, the Node class still exposes an API that
    allows advanced users to control and inspect almost every major step of the
    simulation. For a standard run, users are encouraged to use the higher-level
    `Neurodamus` class instead, which simplifies some of the complexities
    handled by Node.

    The Node class exposes the following public properties:

    - `circuits`: is a :class:`neurodamus.node.CircuitManager` object,
      responsible for managing multiple node and edge managers within the
      simulation.
    - `target_manager`: is a :class:`neurodamus.target_manager.TargetManager`
      object, responsible for managing the targets in the simulation.
    - `stimulus_manager`: is a
      :class:`neurodamus.stimulus_manager.StimulusManager` object, responsible
      for interpreting and instantiating stimulus events.
    - `elec_manager`: The electrode manager, which controls the interaction with
      simulation electrodes.
    - `reports`: A list of Neurodamus Report `hoc` objects, used to generate
      simulation reports.

    Note that, while the Node object owns and manages most of the top-level
    objects in the simulation, the management of cell and synapse objects has
    been delegated to the `Circuits` class, as these are now handled at a lower
    level.

    Technical note:

    - The properties exposed by Node are read-only, with most internal
      attributes being prefixed with an underscore (`_`). Notable internal
      attributes include:

      `self._sonata_circuits`: The SONATA circuits used by the Node
      each represents a node population.

    These details make the Node class versatile and powerful for advanced users
    who need more granular control over the simulation process.
    """

    _default_population = "All"
    """The default population name for e.g. Reports."""

    def __init__(self, config_file, options: dict | None = None):
        """Creates a neurodamus executor
        Args:
            config_file: A Sonata config file
            options: A dictionary of run options typically coming from cmd line
        """
        options = options or {}
        assert isinstance(config_file, str), "`config_file` should be a string"
        assert config_file, "`config_file` cannot be empty"

        if config_file.endswith("BlueConfig"):
            raise ConfigurationError(
                "Legacy format BlueConfig is not supported, please migrate to SONATA config"
            )
        import libsonata

        conf = libsonata.SimulationConfig.from_file(config_file)
        Nd.init(log_filename=conf.output.log_file, log_use_color=options.pop("use_color", True))

        # This is global initialization, happening once, regardless of number of
        # cycles
        log_stage("Setting up Neurodamus configuration")
        self._pc = Nd.pc
        self._spike_vecs = []
        self._spike_populations = []
        Nd.execute("cvode = new CVode()")
        SimConfig.init(config_file, options)

        if SimConfig.use_coreneuron:
            # Instantiate the CoreNEURON artificial cell object which is used to fill up
            # the empty ranks. This need to be done before the circuit is
            # finitialized
            CoreConfig.instantiate_artificial_cell()

        self._run_conf = SimConfig.run_conf
        self._target_manager = TargetManager(self._run_conf)
        self._target_spec = TargetSpec(self._run_conf.get("CircuitTarget"))
        if SimConfig.use_neuron or SimConfig.coreneuron_direct_mode:
            self._sonatareport_helper = Nd.SonataReportHelper(Nd.dt, True)  # noqa: FBT003
        self._sonata_circuits = SimConfig.sonata_circuits
        self._dump_cell_state_gids = get_debug_cell_gids(options)
        self._core_replay_file = ""
        self._is_ngv_run = any(
            c.Engine.__name__ == "NGVEngine" for c in self._sonata_circuits.values() if c.Engine
        )
        self._initial_rss = 0
        self._cycle_i = 0
        self._n_cycles = 1
        self._shm_enabled = False
        self._dry_run_stats = None

        self._reset()

    def _reset(self):
        """Resets internal state for a new simulation cycle.

        Ensures `_run_conf` is a valid dictionary, initializes core attributes,
        and registers global targets and cell managers.

        Note: remember to call Nd.init(...) before to ensure/load neurodamus mods
        """
        if not self._run_conf or not isinstance(self._run_conf, dict):
            raise ValueError("Invalid `_run_conf`: Must be a dictionary for multi-cycle runs.")

        # Init unconditionally
        self._circuits = CircuitManager()
        self._stim_list = None
        self._report_list = None
        self._stim_manager = None
        self._sim_ready = False
        # flag to mark what we already dumped
        self._last_cell_state_dump_t = None

        self._bbss = Nd.BBSaveState()

        # Register the global target and cell manager
        self._target_manager.register_target(self._circuits.global_target)
        self._target_manager.register_cell_manager(self._circuits.global_manager)

    # public 'read-only' properties - object modification on user responsibility
    circuits = property(lambda self: self._circuits)
    target_manager = property(lambda self: self._target_manager)
    stim_manager = property(lambda self: self._stim_manager)
    stims = property(lambda self: self._stim_list)
    reports = property(lambda self: self._report_list)

    def all_circuits(self):
        yield from self._sonata_circuits.values()

    # -
    def load_targets(self):
        """Initialize targets. Nodesets are loaded on demand."""
        for circuit in self.all_circuits():
            log_verbose("Loading targets for circuit %s", circuit.name or "(default)")
            self._target_manager.load_targets(circuit)

    # -
    @mpi_no_errors
    @timeit(name="Compute LB")
    def compute_load_balance(self):
        """In case the user requested load-balance this function instantiates a
        CellDistributor to split cells and balance those pieces across the available CPUs.
        """
        log_stage("Computing Load Balance")
        circuit = None
        for name, circuit in self._sonata_circuits.items():
            if circuit.get("PopulationType") != "virtual":
                logging.info("Activating experimental LB for Sonata circuit '%s'", name)
                break
        if circuit is None:
            logging.warning(
                "Cannot calculate the load balance because no non-virtual circuit is found"
            )
            return None

        if not circuit.CellLibraryFile:
            logging.info(" => No circuit for Load Balancing. Skipping... ")
            return None

        _ = PopulationNodes.offset_freezer()  # Dont offset while in loadbal

        # Info about the cells to be distributed
        target_spec = TargetSpec(circuit.CircuitTarget)
        target = self.target_manager.get_target(target_spec)

        # Check / set load balance mode
        lb_mode = LoadBalance.select_lb_mode(SimConfig, self._run_conf, target)
        if lb_mode == LoadBalanceMode.RoundRobin:
            return None
        if lb_mode == LoadBalanceMode.Memory:
            logging.info("Load Balancing ENABLED. Mode: Memory")
            return self._memory_mode_load_balancing()

        # Build load balancer as per requested options
        node_path = circuit.CellLibraryFile
        pop = target_spec.population
        load_balancer = LoadBalance(lb_mode, node_path, pop, self._target_manager)

        if load_balancer.valid_load_distribution(target_spec):
            logging.info("Load Balancing done.")
            return load_balancer

        logging.info("Could not reuse load balance data. Doing a Full Load-Balance")
        cell_dist = self._circuits.new_node_manager(circuit, self._target_manager, self._run_conf)
        with load_balancer.generate_load_balance(target_spec, cell_dist):
            # Instantiate the circuit cells and synapses to evaluate complexities
            cell_dist.finalize()
            self._circuits.global_manager.finalize()
            SimConfig.update_connection_blocks(self._circuits.alias)
            target_manager = self._target_manager
            self._create_synapse_manager(SynapseRuleManager, circuit, target_manager)

        # reset since we instantiated with RR distribution
        Nd.t = 0.0  # Reset time
        self.clear_model()

        return load_balancer

    def _memory_mode_load_balancing(self):
        filename = f"allocation_r{MPI.size}_c{SimConfig.modelbuilding_steps}.pkl.gz"

        file_exists = ospath.exists(filename)
        MPI.barrier()

        self._dry_run_stats = DryRunStats()
        if file_exists:
            alloc = self._dry_run_stats.import_allocation_stats(filename, self._cycle_i)
        else:
            logging.warning("Allocation file not found. Generating on-the-fly.")

            compute_cell_memory_usage = not Path(DryRunStats._MEMORY_USAGE_FILENAME).exists()
            if not compute_cell_memory_usage:
                self._dry_run_stats.try_import_cell_memory_usage()
            else:
                logging.warning("Cell memory usage file not found. Computing on-the-fly.")
            for circuit in self._sonata_circuits.values():
                if circuit.get("PopulationType") == "biophysical":
                    cell_distributor = CellDistributor(
                        circuit, self._target_manager, self._run_conf
                    )
                    cell_distributor.load_nodes(
                        None,
                        loader_opts={
                            "load_mode": "load_nodes_metype",
                            "dry_run_stats": self._dry_run_stats,
                        },
                    )
                    if compute_cell_memory_usage:
                        cell_distributor.finalize(dry_run_stats_obj=self._dry_run_stats)
            if compute_cell_memory_usage:
                self._dry_run_stats.collect_all_mpi()
                self._dry_run_stats.export_cell_memory_usage()
                # reset since we instantiated
                Nd.t = 0.0  # Reset time
                self.clear_model()

            alloc, _, _ = self._dry_run_stats.distribute_cells_with_validation(
                MPI.size, SimConfig.modelbuilding_steps
            )
        for pop, ranks in alloc.items():
            for rank, gids in ranks.items():
                logging.debug("Population: %s, Rank: %s, Number of GIDs: %s", pop, rank, len(gids))
        return alloc

    # -
    @mpi_no_errors
    @timeit(name="Cell creation")
    def create_cells(self, load_balance=None):
        """Instantiate and distributes the cells of the network.
        Any targets will be updated to know which cells are local to the cpu.
        """
        if SimConfig.dry_run:
            logging.info("Memory usage after inizialization:")
            print_mem_usage()
            self._dry_run_stats = DryRunStats()
            self._dry_run_stats.try_import_cell_memory_usage()
            loader_opts = {"dry_run_stats": self._dry_run_stats}
        else:
            loader_opts = {}

        loader_opts["cycle_i"] = self._cycle_i

        # Check dynamic attributes required before loading cells
        SimConfig.check_cell_requirements(self.target_manager)

        log_stage("LOADING NODES")
        config = SimConfig.cli_options
        if not load_balance:
            logging.info("Load-balance object not present. Continuing Round-Robin...")

        for name, circuit in self._sonata_circuits.items():
            log_stage("Circuit %s", name)
            if config.restrict_node_populations and name not in config.restrict_node_populations:
                logging.warning("Skipped node population (restrict_node_populations)")
                continue
            self._circuits.new_node_manager(
                circuit,
                self._target_manager,
                self._run_conf,
                load_balancer=load_balance,
                loader_opts=loader_opts,
            )

        lfp_weights_file = self._run_conf.get("LFPWeightsPath")
        if lfp_weights_file:
            if SimConfig.use_coreneuron:
                lfp_manager = self._circuits.global_manager._lfp_manager
                cell_managers = self._circuits.global_manager._cell_managers
                population_list = [
                    manager.population_name
                    for manager in cell_managers
                    if manager.population_name is not None
                ]
                lfp_manager.load_lfp_config(lfp_weights_file, population_list)
            else:
                logging.warning("LFP supported only with CoreNEURON.")

        PopulationNodes.freeze_offsets()  # Dont offset further, could change gids

        # Let the cell managers have any final say in the cell objects
        log_stage("FINALIZING CIRCUIT CELLS")

        for cell_manager in self._circuits.all_node_managers():
            log_stage("Circuit %s", cell_manager.circuit_name or "(default)")
            if SimConfig.dry_run:
                cell_manager.finalize(dry_run_stats_obj=self._dry_run_stats)
            else:
                cell_manager.finalize()

        if SimConfig.dry_run:
            self._dry_run_stats.collect_all_mpi()
            self._dry_run_stats.export_cell_memory_usage()
            self._dry_run_stats.estimate_cell_memory()

        # Final bits after we have all cell managers
        self._circuits.global_manager.finalize()
        SimConfig.update_connection_blocks(self._circuits.alias)

    # -
    @mpi_no_errors
    @timeit(name="Synapse creation")
    def create_synapses(self):
        """Create synapses among the cells, handling connections that appear in the config file"""
        log_stage("LOADING CIRCUIT CONNECTIVITY")
        target_manager = self._target_manager
        manager_kwa = {
            "load_offsets": self._is_ngv_run,
            "dry_run_stats": self._dry_run_stats,
        }

        for circuit in self._sonata_circuits.values():
            Engine = circuit.Engine or METypeEngine
            SynManagerCls = Engine.InnerConnectivityCls
            self._create_synapse_manager(SynManagerCls, circuit, target_manager, **manager_kwa)

        MPI.check_no_errors()
        log_stage("Handling projections...")
        for pname, projection in SimConfig.projections.items():
            self._load_projections(pname, projection, **manager_kwa)

        if SimConfig.dry_run:
            self.syn_total_memory = self._dry_run_stats.collect_display_syn_counts()
            return

        log_stage("Configuring connections...")
        for conn_conf in SimConfig.connections.values():
            self._process_connection_configure(conn_conf)

        logging.info("Done, but waiting for all ranks")

    def _create_synapse_manager(self, ctype, conf, *args, **kwargs):
        """Create a synapse manager for intra-circuit connectivity"""
        log_stage("Circuit %s", conf.name or "(default)")
        if not conf.get("nrnPath"):
            logging.info(" => No connectivity set as internal. See projections")
            return

        if SimConfig.cli_options.restrict_connectivity >= 2:
            logging.warning("Skipped connectivity (restrict_connectivity)")
            return

        c_target = TargetSpec(conf.get("CircuitTarget"))
        if c_target.population is None:
            c_target.population = self._circuits.alias.get(conf.name)

        edge_file, *pop = conf.get("nrnPath").split(":")
        edge_pop = pop[0] if pop else None
        src, dst = edge_node_pop_names(edge_file, edge_pop)

        logging.info("Processing edge file %s, pop: %s", edge_file, edge_pop)

        if src and dst and src != dst:
            raise ConfigurationError("Inner connectivity with different populations")

        dst = self.circuits.alias.get(dst, dst)
        if dst not in SimConfig.cli_options.restrict_node_populations:
            logging.warning("Skipped connectivity (restrict_node_populations)")
            return

        manager = self._circuits.get_create_edge_manager(
            ctype, src, dst, c_target, (conf, *args), **kwargs
        )
        if manager.is_file_open:  # Base connectivity
            manager.create_connections()

    def _process_connection_configure(self, conn_conf):
        source_t = TargetSpec(conn_conf["Source"])
        dest_t = TargetSpec(conn_conf["Destination"])
        source_t.population, dest_t.population = self._circuits.unalias_pop_keys(
            source_t.population, dest_t.population
        )
        src_target = self.target_manager.get_target(source_t)
        dst_target = self.target_manager.get_target(dest_t)
        # Loop over population pairs
        for src_pop in src_target.population_names:
            for dst_pop in dst_target.population_names:
                # Loop over all managers having connections between the populations
                for conn_manager in self._circuits.get_edge_managers(src_pop, dst_pop):
                    logging.debug("Using connection manager: %s", conn_manager)
                    conn_manager.configure_connections(conn_conf)

    @mpi_no_errors
    def _load_projections(self, pname, projection, **kw):
        """Check for Projection blocks"""
        target_manager = self._target_manager
        # None, GapJunctions, NeuroGlial, NeuroModulation...
        ptype = projection.get("Type")
        ptype_cls = EngineBase.connection_types.get(ptype)
        source_t = TargetSpec(projection.get("Source"))
        dest_t = TargetSpec(projection.get("Destination"))

        if SimConfig.cli_options.restrict_connectivity >= 1:
            logging.warning("Skipped projections %s->%s (restrict_connectivity)", source_t, dest_t)
            return

        if not ptype_cls:
            raise RuntimeError(f"No Engine to handle connectivity of type '{ptype}'")

        ppath, *pop_name = projection["Path"].split(":")
        edge_pop_name = pop_name[0] if pop_name else None

        logging.info("Processing Edge file: %s", ppath)

        # Update the target spec with the actual populations
        src_pop, dst_pop = edge_node_pop_names(
            ppath, edge_pop_name, source_t.population, dest_t.population
        )
        source_t.population, dest_t.population = self._circuits.unalias_pop_keys(src_pop, dst_pop)
        src_target = self.target_manager.get_target(source_t)
        dst_target = self.target_manager.get_target(dest_t)

        # If the src_pop is not a known node population, allow creating a Virtual one
        src_populations = src_target.population_names or [source_t.population]

        for src_pop in src_populations:
            for dst_pop in dst_target.population_names:
                logging.info(" * %s (Type: %s, Src: %s, Dst: %s)", pname, ptype, src_pop, dst_pop)
                conn_manager = self._circuits.get_create_edge_manager(
                    ptype_cls,
                    src_pop,
                    dst_pop,
                    source_t,
                    (projection, target_manager),
                    **kw,  # args to ptype_cls if creating
                )
                logging.debug("Using connection manager: %s", conn_manager)
                proj_source = ":".join([ppath, *pop_name])
                conn_manager.open_edge_location(proj_source, projection, src_name=src_pop)
                conn_manager.create_connections(source_t.name, dest_t.name)

    @mpi_no_errors
    @timeit(name="Enable Stimulus")
    def enable_stimulus(self):
        """Iterate over any stimulus defined in the config file given to the simulation
        and instantiate them.
        This passes the raw text in field/value pairs to a StimulusManager object to interpret the
        text and instantiate an actual stimulus object.
        """
        if Feature.Stimulus not in SimConfig.cli_options.restrict_features:
            logging.warning("Skipped Stimulus (restrict_features)")
            return

        log_stage("Stimulus Apply.")

        # for each stimulus defined in the config file, request the StimulusManager to
        # instantiate
        self._stim_manager = StimulusManager(self._target_manager)

        for stim in SimConfig.stimuli:
            if stim.get("Mode") == "Extracellular":
                raise ConfigurationError("input_type extracellular_stimulation is not supported")
            target_spec = TargetSpec(stim.get("Target"))
            stim_name = stim["Name"]
            stim_pattern = stim["Pattern"]
            if stim_pattern == "SynapseReplay":
                continue  # Handled by enable_replay
            logging.info(
                " * [STIM] %s (%s): -> %s",
                stim_name,
                stim_pattern,
                target_spec,
            )
            self._stim_manager.interpret(target_spec, stim)

    # -
    @mpi_no_errors
    def enable_replay(self):
        """Activate replay according to config file. Call before connManager.finalize"""
        if Feature.Replay not in SimConfig.cli_options.restrict_features:
            logging.warning("Skipped Replay (restrict_features)")
            return

        log_stage("Handling Replay")

        if SimConfig.use_coreneuron and bool(self._core_replay_file):
            logging.info(" -> [REPLAY] Reusing stim file from previous cycle")
            return

        for stim in SimConfig.stimuli:
            if stim.get("Pattern") != "SynapseReplay":
                continue
            target = stim["Target"]
            source = stim.get("Source")
            stim_name = stim["Name"]

            #  - delay: Spike replays are suppressed until a certain time
            delay = stim.get("Delay", 0.0)
            logging.info(
                " * [SYN REPLAY] %s -> %s (delay: %d)",
                stim_name,
                target,
                delay,
            )
            self._enable_replay(source, target, stim, delay=delay)

    # -
    def _enable_replay(
        self, source, target, stim_conf, tshift=0.0, delay=0.0, connectivity_type=None
    ):
        ptype_cls = EngineBase.connection_types.get(connectivity_type)
        src_target = self.target_manager.get_target(source)
        dst_target = self.target_manager.get_target(target)

        if SimConfig.restore_coreneuron:
            pop_offsets, alias_pop, _virtual_pop_offsets = CircuitManager.read_population_offsets()

        for src_pop in src_target.population_names:
            try:
                log_verbose("Loading replay spikes for population '%s'", src_pop)
                spike_manager = SpikeManager(stim_conf["SpikeFile"], tshift, src_pop)  # Disposable
            except MissingSpikesPopulationError:
                logging.info("  > No replay for src population: '%s'", src_pop)
                continue

            for dst_pop in dst_target.population_names:
                src_pop_str, dst_pop_str = src_pop or "(base)", dst_pop or "(base)"

                if SimConfig.restore_coreneuron:  # Node and Edges managers not initialized
                    src_pop_offset = (
                        pop_offsets[src_pop]
                        if src_pop in pop_offsets
                        else pop_offsets[alias_pop[src_pop]]
                    )
                else:
                    conn_manager = self._circuits.get_edge_manager(src_pop, dst_pop, ptype_cls)
                    if not conn_manager and SimConfig.cli_options.restrict_connectivity >= 1:
                        continue
                    assert conn_manager, f"Missing edge manager for {src_pop_str} -> {dst_pop_str}"
                    src_pop_offset = conn_manager.src_pop_offset

                logging.info(
                    "=> Population pathway %s -> %s. Source offset: %d",
                    src_pop_str,
                    dst_pop_str,
                    src_pop_offset,
                )
                conn_manager.replay(spike_manager, source, target, delay)

    # -
    @mpi_no_errors
    @timeit(name="Enable Modifications")
    def enable_modifications(self):
        """Iterate over any Modification blocks read from the config file and apply them to the
        network. The steps needed are more complex than NeuronConfigures, so the user should not be
        expected to write the hoc directly, but rather access a library of already available mods.
        """
        # mod_mananger gets destroyed when function returns (not required)
        # mod_manager = Nd.ModificationManager(self._target_manager.hoc)
        log_stage("Enabling modifications...")

        mod_manager = ModificationManager(self._target_manager)
        for name, mod_info in SimConfig.modifications.items():
            target_spec = TargetSpec(mod_info["Target"])
            logging.info(" * [MOD] %s: %s -> %s", name, mod_info["Type"], target_spec)
            mod_manager.interpret(target_spec, mod_info)

    # Reporting
    class ReportParams(typing.NamedTuple):
        rep_type: str
        name: str
        report_on: str
        unit: str
        format: str
        dt: float
        start: float
        end: float
        output_dir: str
        scaling: str

    def write_and_get_population_offsets(self) -> tuple[dict, dict, dict]:
        """Retrieve population offsets from the circuit or restore them,
        write the offsets, and return them.

        Returns:
            tuple[dict, dict, dict]:
                - pop_offsets: Mapping of population names to GID offsets.
                - alias_pop: Mapping of population aliases to population names.
                - virtual_pop_offsets: Mapping of virtual population names to offsets.
        """
        if self._circuits.initialized():
            pop_offsets, alias_pop = self._circuits.get_population_offsets()
            virtual_pop_offsets = self._circuits.get_virtual_population_offsets()
        else:
            # restore way
            pop_offsets, alias_pop, virtual_pop_offsets = CircuitManager.read_population_offsets()
        self._circuits.write_population_offsets(
            pop_offsets, alias_pop, virtual_pop_offsets=virtual_pop_offsets
        )
        return pop_offsets, alias_pop, virtual_pop_offsets

    # @mpi_no_errors - not required since theres a call inside before make_comm()
    @timeit(name="Enable Reports")
    def enable_reports(self):  # noqa: C901, PLR0912
        """Iterate over reports defined in the config file and instantiate them."""
        log_stage("Reports Enabling")

        errors = []
        # filter: only the enabled ones
        reports_conf = {name: conf for name, conf in SimConfig.reports.items() if conf["Enabled"]}
        self._report_list = []

        pop_offsets, alias_pop, _virtual_pop_offsets = self.write_and_get_population_offsets()
        pop_offsets_alias = pop_offsets, alias_pop

        if SimConfig.use_coreneuron:
            if SimConfig.restore_coreneuron:
                # we copy it first. We will proceed to modify
                # it in update_report_config later in one go
                Path(CoreConfig.report_config_file_save).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(
                    CoreConfig.report_config_file_restore, CoreConfig.report_config_file_save
                )
            else:
                CoreConfig.write_report_count(len(reports_conf))

        # necessary for restore: we need to update the various reports tend
        # we can do it in one go later
        substitutions = {}
        for rep_name, rep_conf in reports_conf.items():
            target_spec = TargetSpec(rep_conf["Target"])
            target = self._target_manager.get_target(target_spec)

            # Build final config. On errors log, stop only after all reports processed
            try:
                rep_params = self._report_build_params(rep_name, rep_conf)
            except Exception as e:
                logging.exception("Error setting up report parameters '%s'", rep_name)
                errors.append(("_report_setup", rep_name, e))
                continue

            if SimConfig.restore_coreneuron:
                substitutions[rep_params.name, target_spec.name] = rep_params.end
                continue  # we dont even need to initialize reports

            if SimConfig.use_coreneuron:
                self._coreneuron_write_report_config(rep_conf, target, rep_params)

            has_gids = len(self._circuits.global_manager.get_final_gids()) > 0
            report = (
                create_report(params=rep_params, use_coreneuron=SimConfig.use_coreneuron)
                if has_gids
                else None
            )

            # With coreneuron direct mode, enable fast membrane current calculation
            # for i_membrane
            if (
                SimConfig.coreneuron_direct_mode and "i_membrane" in rep_params.report_on
            ) or rep_params.rep_type == "lfp":
                Nd.cvode.use_fast_imem(1)

            if not SimConfig.use_coreneuron or rep_params.rep_type == "synapse":
                try:
                    self._report_setup(report, rep_conf, target, rep_params.rep_type)
                except Exception as e:
                    logging.exception("Error setting up report '%s'", rep_name)
                    errors.append(("_report_setup", rep_name, e))
                    continue

            self._report_list.append(report)

        if SimConfig.restore_coreneuron:
            CoreConfig.update_report_config(substitutions)

        if errors:
            raise ReportsCumulativeError(errors)

        MPI.check_no_errors()

        if not SimConfig.restore_coreneuron:
            self._reports_init(pop_offsets_alias)

    def _report_build_params(self, rep_name, rep_conf):
        """Build and validate report parameters from configuration.

        Ensures report timing and settings are consistent with simulation constraints,
        raises ReportSetupError on invalid configurations (e.g., missing LFP setup,
        invalid time ranges, or incompatible time steps).

        Returns:
            ReportParams: A populated report parameters object.
        """
        sim_end = self._run_conf["Duration"]
        rep_type = rep_conf["Type"]
        start_time = rep_conf["StartTime"]
        end_time = rep_conf.get("EndTime", sim_end)
        rep_dt = rep_conf["Dt"]
        rep_format = rep_conf["Format"]

        lfp_disabled = not self._circuits.global_manager._lfp_manager._lfp_file
        if rep_type == "lfp" and lfp_disabled:
            raise ReportSetupError(
                "LFP report setup failed: electrodes file may be missing "
                "or simulator is not set to CoreNEURON."
            )
        if rep_type == "compartment_set" and SimConfig.use_coreneuron:
            raise ReportSetupError(
                "Compartment set reports are not supported with CoreNEURON at the moment."
            )
        logging.info(
            " * %s (Type: %s, Target: %s, Dt: %f)",
            rep_name,
            rep_type,
            rep_conf["Target"],
            rep_dt,
        )

        if Nd.t > 0:
            start_time += Nd.t
            end_time += Nd.t
        end_time = min(end_time, sim_end)
        if start_time > end_time:
            raise ReportSetupError(
                f"Invalid report configuration: end time ({end_time}) is before "
                f"start time ({start_time})."
            )

        if rep_dt < Nd.dt:
            raise ReportSetupError(
                f"Invalid report configuration: report dt ({rep_dt}) is smaller than "
                f"simulation dt ({Nd.dt})."
            )

        rep_target = TargetSpec(rep_conf["Target"])
        population_name = (
            rep_target.population or self._target_spec.population or self._default_population
        )
        log_verbose("Report on Population: %s, Target: %s", population_name, rep_target.name)

        report_on = rep_conf["ReportOn"]
        return self.ReportParams(
            rep_type,  # rep type is case sensitive !!
            os.path.basename(rep_conf.get("FileName", rep_name)),
            report_on,
            rep_conf["Unit"],
            rep_format,
            rep_dt,
            start_time,
            end_time,
            SimConfig.output_root,
            rep_conf.get("Scaling"),
        )

    @staticmethod
    @run_only_rank0
    def _coreneuron_write_report_config(rep_conf, target, rep_params):
        """Configure CoreNEURON reporting based on the provided configuration.

        Computes the target type (if "Sections" and "Compartments" are specified)
        and writes the report configuration to CoreConfig.

        Args:
            rep_conf (dict): Report configuration with target, sections, and compartments.
            target (Target): Target object with name and GIDs.
            rep_params (ReportParams): Report parameters including name, type, and variables.
        """
        target_spec = TargetSpec(rep_conf["Target"])

        # For restore case with no change in reporting, we can directly update the end time.
        # Note: If different reports are needed during restore, this workflow
        # needs to be adapted.

        # for sonata config, compute target_type from user inputs
        if "Sections" in rep_conf and "Compartments" in rep_conf:

            def _compute_corenrn_target_type(section_type, compartment_type):
                sections = ["all", "soma", "axon", "dend", "apic"]
                compartments = ["center", "all"]
                if section_type == "all":  # for "all sections", support only target_type=0
                    return 0
                # 0=Compartment, Section { 2=Soma, 3=Axon, 4=Dendrite, 5=Apical,
                # 6=SomaAll ... }
                return sections.index(section_type) + 1 + 4 * compartments.index(compartment_type)

            section_type = rep_conf.get("Sections")
            compartment_type = rep_conf.get("Compartments")
            target_type = _compute_corenrn_target_type(section_type, compartment_type)

        reporton_comma_separated = ",".join(rep_params.report_on.split())
        core_report_params = (
            rep_params.name,
            target_spec.name,
            rep_params.rep_type,
            reporton_comma_separated,
            *rep_params[3:5],
            *(target_type,),
            *rep_params[5:8],
            *(target.get_gids(), SimConfig.report_buffer_size),
        )
        CoreConfig.write_report_config(*core_report_params)

    def _report_setup(self, report, rep_conf, target, rep_type):
        if report is None:
            return
        global_manager = self._circuits.global_manager

        sum_currents_into_soma = False
        if rep_type == "compartment_set":
            compartment_set = rep_conf.get("CompartmentSet")
            points = self._target_manager.get_point_list(target, compartment_set=compartment_set)
        else:
            # Go through the target members, one cell at a time. We give a cell reference
            # For summation targets - check if we were given a Cell target because we really
            # want all points of the cell which will ultimately be collapsed to a single
            # value on the soma. Otherwise, get target points as normal.
            sections = rep_conf.get("Sections")
            compartments = rep_conf.get("Compartments")

            sum_currents_into_soma = sections == "soma" and compartments == "center"
            # In case of summation in the soma, we need all points anyway
            if sum_currents_into_soma and rep_type == "summation":
                sections = "all"
                compartments = "all"
            points = self._target_manager.get_point_list(
                target, sections=sections, compartments=compartments
            )
        for point in points:
            gid = point.gid
            pop_name, pop_offset = global_manager.getPopulationInfo(gid)
            cell = global_manager.get_cell(gid)
            spgid = global_manager.getSpGid(gid)

            report.register_gid_section(
                cell, point, spgid, pop_name, pop_offset, sum_currents_into_soma
            )

    def _reports_init(self, pop_offsets_alias):
        pop_offsets = pop_offsets_alias[0]
        if SimConfig.use_coreneuron:
            # write spike populations
            if hasattr(CoreConfig, "write_population_count"):
                # Do not count populations with None pop_name
                pop_count = len(pop_offsets) - 1 if None in pop_offsets else len(pop_offsets)
                CoreConfig.write_population_count(pop_count)
            for pop_name, offset in pop_offsets.items():
                if pop_name is not None:
                    CoreConfig.write_spike_population(pop_name or "All", offset)
            spike_path = self._run_conf.get("SpikesFile")
            if spike_path is not None:
                # Get only the spike file name
                file_name = spike_path.split("/")[-1]
            CoreConfig.write_spike_filename(file_name)
        else:
            # once all reports are created, we finalize the communicator for any reports
            self._sonatareport_helper.set_max_buffer_size_hint(SimConfig.report_buffer_size)
            self._sonatareport_helper.make_comm()
            self._sonatareport_helper.prepare_datasets()

    # -
    @mpi_no_errors
    def sim_init(self, corenrn_gen=None, **sim_opts):
        """Finalize the model and prepare to run simulation.

        After finalizing the model, will eventually write coreneuron config
        and initialize the neuron simulation if applicable.

        Args:
            corenrn_gen: Whether to generate coreneuron config. Default: None (if required)
            sim_opts - override _finalize_model options. E.g. spike_compress
        """
        if self._sim_ready:
            return self._pc

        if not len(self._circuits.all_node_managers()):
            raise RuntimeError("No CellDistributor was initialized. Please create a circuit.")

        self._finalize_model(**sim_opts)

        if corenrn_gen is None:
            corenrn_gen = SimConfig.use_coreneuron
        if corenrn_gen:
            self._coreneuron_configure_datadir(
                corenrn_restore=False, coreneuron_direct_mode=SimConfig.coreneuron_direct_mode
            )
            self._coreneuron_write_sim_config()

        if SimConfig.use_neuron or SimConfig.coreneuron_direct_mode:
            self._sim_init_neuron()

        assert not (SimConfig.use_neuron and SimConfig.use_coreneuron)
        if SimConfig.use_neuron:
            self.dump_cell_states()

        self._sim_ready = True
        return self._pc

    # -
    @mpi_no_errors
    @timeit(name="Model Finalized")
    def _finalize_model(self, spike_compress=3):
        """Set up simulation run parameters and initialization.

        Handles setup_transfer, spike_compress, _record_spikes, stdinit, timeout
        Args:
            spike_compress: The spike_compress() parameters (tuple or int)
        """
        logging.info("Preparing to run simulation...")
        for mgr in self._circuits.all_node_managers():
            mgr.pre_stdinit()

        is_save_state = SimConfig.save or SimConfig.restore
        self._pc.setup_transfer()

        if spike_compress and not is_save_state and not self._is_ngv_run:
            # multisend 13 is combination of multisend(1) + two_phase(8) + two_intervals(4)
            # to activate set spike_compress=(0, 0, 13)
            if SimConfig.loadbal_mode == LoadBalanceMode.Memory:
                logging.info("Disabling spike compression for Memory Load Balance")
                spike_compress = False
            if not isinstance(spike_compress, tuple):
                spike_compress = (spike_compress, 1, 0)
            self._pc.spike_compress(*spike_compress)

        # LFP calculation requires WholeCell balancing and extracellular mechanism.
        # This is incompatible with efficient caching atm AND Incompatible with
        # mcd & Glut
        if not self._is_ngv_run:
            Nd.cvode.cache_efficient("ElectrodesPath" not in self._run_conf)
        self._pc.set_maxstep(4)
        with timeit(name="stdinit"):
            Nd.stdinit()

    # -
    def _sim_init_neuron(self):
        # === Neuron specific init ===
        restore_path = SimConfig.restore

        # create a spike_id vector which stores the pairs for spikes and timings for
        # every engine
        for cell_manager in self._circuits.all_node_managers():
            if cell_manager.population_name is not None:
                self._spike_populations.append(
                    (cell_manager.population_name, cell_manager.local_nodes.offset)
                )
                self._spike_vecs.append(cell_manager.record_spikes() or (Nd.Vector(), Nd.Vector()))

        self._pc.timeout(200)  # increase by 10x

        if restore_path:
            with timeit(name="restoretime"):
                logging.info("Restoring state...")
                self._stim_manager.saveStatePreparation(self._bbss)
                self._bbss.vector_play_init()
                self._restart_events()  # On restore the event queue is cleared
                return  # Upon restore sim is ready

    # -
    def _restart_events(self):
        logging.info("Restarting connections events (Replay and Spont Minis)")
        for syn_manager in self._circuits.all_synapse_managers():
            syn_manager.restart_events()

    @contextmanager
    def _coreneuron_ensure_all_ranks_have_gids(self, corenrn_data):
        local_gid_count = sum(
            len(manager.local_nodes) for manager in self._circuits.all_node_managers()
        )
        if local_gid_count > 0:
            yield
            return

        # Create a dummy cell manager with node_pop = None
        # which holds a fake node with a fake population "zzz" to get an unused gid.
        # coreneuron fails if this edge case is reached multiple times as we
        # try to add twice the same gid. pop "zzz" is reserved to be used
        # exclusively for handling cases where no real GIDs are assigned to
        # a rank, ensuring that CoreNeuron does not crash due to missing GIDs.
        log_verbose("Creating fake gid for CoreNeuron")
        assert not PopulationNodes.get("zzz"), "Population 'zzz' is reserved "
        "for handling empty GID ranks and should not be used elsewhere."
        pop_group = PopulationNodes.get("zzz", create=True)
        fake_gid = pop_group.offset + 1 + MPI.rank
        # Add the fake cell to a dummy manager
        dummy_cell_manager = CellDistributor(
            circuit_conf=make_circuit_config({"CellLibraryFile": "<NONE>"}),
            target_manager=self._target_manager,
        )
        dummy_cell_manager.load_artificial_cell(fake_gid, CoreConfig.artificial_cell_object)
        yield

        # register_mapping() doesn't work for this artificial cell as somatic attr is
        # missing, so create a dummy mapping file manually, required for reporting
        cur_files = glob.glob(f"{corenrn_data}/*_3.dat")
        example_mapfile = cur_files[0]
        with open(example_mapfile, "rb") as f_mapfile:
            # read the version from the existing mapping file generated by coreneuron
            coredata_version = f_mapfile.readline().rstrip().decode("ascii")

        mapping_file = Path(corenrn_data, f"{fake_gid}_3.dat")
        if not mapping_file.is_file():
            mapping_file.write_text(f"{coredata_version}\n0\n", encoding="utf-8")

    def _coreneuron_configure_datadir(self, corenrn_restore, coreneuron_direct_mode):
        """Configures the CoreNEURON data directory and handles shared memory (SHM) setup.

        - Creates the data directory if it doesn't exist.
        - If in direct mode, returns immediately since the default behavior is fine.
        - If restoring, skips the setup.
        - If not restoring, checks if SHM should be enabled based on available memory,
          and sets up symlinks for CoreNEURON's necessary files in SHM.

        Args:
            corenrn_restore (bool): Flag indicating if CoreNEURON is in restore mode.
            coreneuron_direct_mode (bool): Flag indicating if direct mode is enabled.
        """
        corenrn_datadir = SimConfig.coreneuron_datadir_path(create=True)
        if coreneuron_direct_mode:
            return
        corenrn_datadir_shm = SHMUtil.get_datadir_shm(corenrn_datadir)

        # Clean-up any previous simulations in the same output directory
        if self._cycle_i == 0 and corenrn_datadir_shm:
            rmtree(corenrn_datadir_shm)

        # Ensure that we have a folder in /dev/shm (i.e., 'SHMDIR' ENV variable)
        if SimConfig.cli_options.enable_shm and not corenrn_datadir_shm:
            logging.warning("Unknown SHM directory for model file transfer in CoreNEURON.")
        # Try to configure the /dev/shm folder as the output directory for the files
        elif (
            self._cycle_i == 0
            and not corenrn_restore
            and (SimConfig.cli_options.enable_shm and SimConfig.delete_corenrn_data)
        ):
            # Check for the available memory in /dev/shm and estimate the RSS by multiplying
            # the number of cycles in the multi-step model build with an approximate
            # factor
            mem_avail = SHMUtil.get_mem_avail()
            shm_avail = SHMUtil.get_shm_avail()
            initial_rss = self._initial_rss
            current_rss = SHMUtil.get_node_rss()
            factor = SHMUtil.get_shm_factor()
            rss_diff = (current_rss - initial_rss) if initial_rss < current_rss else current_rss
            # 'rss_diff' prevents <0 estimates
            rss_req = int(rss_diff * self._n_cycles * factor)

            # Sync condition value with all ranks to ensure that all of them can use
            # /dev/shm
            shm_possible = (rss_req < shm_avail) and (rss_req < mem_avail)
            if MPI.allreduce(int(shm_possible), MPI.SUM) == MPI.size:
                logging.info("SHM file transfer mode for CoreNEURON enabled")

                # Create SHM folder and links to GPFS for the global data structures
                os.makedirs(corenrn_datadir_shm, exist_ok=True)

                # Important: These three files must be available on every node, as they are shared
                #            across all of the processes. The trick here is to fool NEURON into
                #            thinking that the files are written in /dev/shm, but they are actually
                #            written on GPFS. The workflow is identical, meaning that rank 0 writes
                #            the content and every other rank reads it afterwards in CoreNEURON.
                for filename in ("bbcore_mech.dat", "files.dat", "globals.dat"):
                    path = os.path.join(corenrn_datadir, filename)
                    path_shm = os.path.join(corenrn_datadir_shm, filename)

                    try:
                        os.close(os.open(path, os.O_CREAT))
                        os.symlink(path, path_shm)
                    except FileExistsError:
                        pass  # Ignore if other process has already created it

                # Update the flag to confirm the configuration
                self._shm_enabled = True
            else:
                logging.warning(
                    "Unable to utilize SHM for model file transfer in CoreNEURON. "
                    "Increase the number of nodes to reduce the memory footprint "
                    "(Current use node: %d MB / SHM Limit: %d MB / Mem. Limit: %d MB)",
                    (rss_req >> 20),
                    (shm_avail >> 20),
                    (mem_avail >> 20),
                )
        _SimConfig.coreneuron_datadir = (
            corenrn_datadir if not self._shm_enabled else corenrn_datadir_shm
        )

    # -
    @timeit(name="corewrite")
    def _coreneuron_write_sim_config(self, corenrn_restore=False):
        log_stage("Dataset generation for CoreNEURON")

        if not corenrn_restore:
            CompartmentMapping(self._circuits.global_manager).register_mapping()
            if not SimConfig.coreneuron_direct_mode:
                with self._coreneuron_ensure_all_ranks_have_gids(CoreConfig.datadir):
                    self._pc.nrnbbcore_write(CoreConfig.datadir)
                    MPI.barrier()  # wait for all ranks to finish corenrn data generation

        prcellgid = self._dump_cell_state_gids[0] if self._dump_cell_state_gids else -1
        if self._dump_cell_state_gids and len(self._dump_cell_state_gids) > 1:
            logging.warning(
                "Multiple cell state GIDs provided. Using the first one: %d",
                self._dump_cell_state_gids[0],
            )
        CoreConfig.write_sim_config(
            Nd.tstop,
            Nd.dt,
            prcellgid,
            getattr(SimConfig, "celsius", 34.0),
            getattr(SimConfig, "v_init", -65.0),
            self._core_replay_file,
            SimConfig.rng_info.getGlobalSeed(),
            int(SimConfig.cli_options.model_stats),
            int(self._run_conf["EnableReports"]),
        )
        # Wait for rank0 to write the sim config file
        MPI.barrier()
        logging.info(" => Dataset written to '%s'", CoreConfig.datadir)

    # -
    def run_all(self):
        """Run the whole simulation according to the simulation config file"""
        if not self._sim_ready:
            self.sim_init()

        timings = None
        if SimConfig.use_neuron:
            timings = self._run_neuron()
            self.sonata_spikes()
        if SimConfig.use_coreneuron:
            print_mem_usage()
            if not SimConfig.coreneuron_direct_mode:
                self.clear_model(avoid_clearing_queues=False)
            self._run_coreneuron()
            if SimConfig.coreneuron_direct_mode:
                self.sonata_spikes()
        return timings

    # -
    @return_neuron_timings
    def _run_neuron(self):
        if MPI.rank == 0:
            _ = SimulationProgress()
        self.solve()
        logging.info("Simulation finished.")

    @staticmethod
    def _run_coreneuron():
        logging.info("Launching simulation with CoreNEURON")
        CoreConfig.psolve_core(
            SimConfig.coreneuron_direct_mode,
        )

    def _sim_event_handlers(self, tstart, tstop):
        """Create handlers for "in-simulation" events, like activating delayed
        connections, execute Save-State, etc
        """
        events = defaultdict(list)  # each key (time) points to a list of handlers

        if SimConfig.save:
            tsave = SimConfig.tstop  # Consider 0 as the end too!
            save_f = self._create_save_handler()
            events[tsave].append(save_f)

        event_list = [(t, events[t]) for t in sorted(events)]
        return event_list

    # -
    def _create_save_handler(self):
        @timeit(name="savetime")
        def save_f():
            """Function that saves the current simulation state:
            syncs MPI, saves stimuli, flushes reports, clears the model,
            and logs progress.
            """
            logging.info("Saving State... (t=%f)", SimConfig.tstop)
            MPI.barrier()
            self._stim_manager.saveStatePreparation(self._bbss)
            log_verbose("SaveState Initialization Done")

            # If event at the end of the sim we can actually clearModel()
            # before savestate()
            log_verbose("Clearing model prior to final save")
            self._sonatareport_helper.flush()

            self.clear_model()
            logging.info(" => Save done successfully")

        return save_f

    # -
    @mpi_no_errors
    @timeit(name="psolve")
    def solve(self, tstop=None):
        """Call solver with a given stop time (default: whole interval).
        Be sure to have sim_init()'d the simulation beforehand
        """
        if not self._sim_ready:
            raise ConfigurationError("Initialize simulation first")

        tstart = Nd.t
        tstop = tstop or Nd.tstop
        event_list = self._sim_event_handlers(tstart, tstop)

        # NOTE: _psolve_loop is called among events in order to eventually split long
        # simulation blocks, where one or more report flush(es) can happen. It is a simplified
        # design relatively to the original version where the report checkpoint would not happen
        # before the checkpoint timeout (25ms default). However there shouldn't be almost any
        # performance penalty since the simulation is already halted between events.

        logging.info("Running simulation until t=%d ms", tstop)
        t = tstart  # default if there are no events
        for t, events in event_list:
            self._psolve_loop(t)
            for event in events:
                event()
            self.dump_cell_states()
        # Run until the end
        if t < tstop:
            self._psolve_loop(tstop)
            self.dump_cell_states()

        # Final flush
        self._sonatareport_helper.flush()

    # psolve_loop: There was an issue where MPI collective routines for reporting and spike exchange
    # are mixed such that some cpus are blocked waiting to complete reporting while others to
    # finish spike exchange. As a work-around, periodically halt simulation and flush reports
    # Default is 25 ms / cycle
    def _psolve_loop(self, tstop):
        cur_t = round(Nd.t, 2)  # fp innnacuracies could lead to infinitesimal loops
        buffer_t = SimConfig.buffer_time
        for _ in range(math.ceil((tstop - cur_t) / buffer_t)):
            next_flush = min(tstop, cur_t + buffer_t)
            self._pc.psolve(next_flush)
            cur_t = next_flush
        Nd.t = cur_t

    # -
    @mpi_no_errors
    def clear_model(self, avoid_creating_objs=False, avoid_clearing_queues=True):
        """Clears appropriate lists and other stored references.
        For use with intrinsic load balancing. After creating and evaluating the network using
        round robin distribution, we want to clear the cells and synapses in order to have a
        clean slate on which to instantiate the balanced cells.
        """
        logging.info("Clearing model")
        self._pc.gid_clear()
        self._target_manager.clear_simulation_data()

        if not avoid_creating_objs and SimConfig.use_neuron and self._sonatareport_helper:
            self._sonatareport_helper.clear()

        # Reset vars
        self._reset()

        # Clear BBSaveState
        self._bbss.ignore()

        # Shrink ArrayPools holding mechanism's data in NEURON
        pool_shrink()

        # Free event queues in NEURON
        if not avoid_clearing_queues:
            free_event_queues()

        # Garbage collect all Python objects without references
        gc.collect()

        # Finally call malloc_trim to return all the freed pages back to the OS
        trim_memory()
        print_mem_usage()

    # -------------------------------------------------------------------------
    #  output
    # -------------------------------------------------------------------------

    def sonata_spikes(self):
        """Write the spike events that occured on each node into a single output SONATA file."""
        output_root = SimConfig.output_root_path(create=True)
        if hasattr(self._sonatareport_helper, "create_spikefile"):
            # Write spike report for multiple populations if exist
            spike_path = self._run_conf.get("SpikesFile")

            # Get only the spike file name
            file_name = spike_path.split("/")[-1] if spike_path is not None else "out.h5"

            # create a sonata spike file
            self._sonatareport_helper.create_spikefile(output_root, file_name)
            # write spikes per population
            for (population, population_offset), (spikevec, idvec) in zip(
                self._spike_populations, self._spike_vecs
            ):
                extra_args = (
                    (population, population_offset) if population else ("All", population_offset)
                )
                self._sonatareport_helper.add_spikes_population(spikevec, idvec, *extra_args)
            # write all spike populations
            self._sonatareport_helper.write_spike_populations()
            # close the spike file
            self._sonatareport_helper.close_spikefile()
        else:
            # fallback: write spike report with one single population "ALL"
            logging.warning(
                "Writing spike reports with multiple populations is not supported. "
                "If needed, please update to a newer version of neurodamus."
            )
            population = self._target_spec.population or "All"
            extra_args = (population,)
            self._sonatareport_helper.write_spikes(spikevec, idvec, output_root, *extra_args)

    def dump_cell_states(self):
        """Dump the _pr_cell_gid cell state if not already done

        We assume that the parallel context is ready. Thus, This function should
        not be called if coreNeuron is employed and we are not at t=0.0.
        """
        assert SimConfig.use_neuron, "This function can work only with Neuron. Use sim.conf to "
        "instruct coreNeuron to dump a cell state instead."
        if not self._dump_cell_state_gids:
            return
        if self._last_cell_state_dump_t == Nd.t:  # avoid duplicating output
            return

        for i in self._dump_cell_state_gids:
            log_verbose("Dumping info about cell %d", i)

            self._pc.prcellstate(i, f"py_Neuron_t{Nd.t}")

        self._last_cell_state_dump_t = Nd.t

    @staticmethod
    @run_only_rank0
    def coreneuron_cleanup():
        """Clean coreneuron save files after running"""
        data_folder = Path(CoreConfig.datadir)
        logging.info("Deleting intermediate data in %s", data_folder)
        assert data_folder.is_dir(), "Data folder must be a directory"
        if data_folder.is_symlink():
            # in restore, coreneuron data is a symbolic link
            data_folder.unlink()
        else:
            rmtree(data_folder)

        build_path = Path(SimConfig.build_path())
        if build_path.exists():
            shutil.rmtree(build_path)

        sim_conf = Path(CoreConfig.sim_config_file)
        assert not sim_conf.exists()

        report_file = Path(CoreConfig.report_config_file_save)
        assert not report_file.exists()

    def cleanup(self):
        """Have the compute nodes wrap up tasks before exiting."""
        # MemUsage constructor will do MPI communications
        print_mem_usage()

        # Coreneuron runs clear the model before starting
        if not SimConfig.use_coreneuron or SimConfig.simulate_model is False:
            self.clear_model(avoid_creating_objs=True)

        if SimConfig.delete_corenrn_data and not SimConfig.save and not SimConfig.dry_run:
            with timeit(name="Delete corenrn data"):
                self.coreneuron_cleanup()
                MPI.barrier()

    @staticmethod
    @run_only_rank0
    def move_dumpcellstates_to_output_root():
        """Check for .corenrn or .nrn files in the current directory
        and move them to CoreConfig.output_root_path(create=True).
        """
        current_dir = Path(".")
        output_root = Path(SimConfig.output_root_path(create=True))

        # Iterate through files in the current directory
        for file in current_dir.iterdir():
            if file.suffix in {".corenrn", ".nrn", ".nrndat"}:
                shutil.move(str(file), output_root / file.name)
                logging.info("Moved %s to %s", file.name, output_root)


class Neurodamus(Node):
    """A high level interface to Neurodamus"""

    def __init__(self, config_file, auto_init=True, logging_level=None, **user_opts):
        """Creates and initializes a neurodamus run node

        As part of Initiazation it calls:
         * load_targets
         * compute_load_balance
         * Build the circuit (cells, synapses, GJs)
         * Add stimulus & replays
         * Activate reports if requested

        Args:
            config_file: The simulation config recipe file
            logging_level: (int) Redefine the global logging level.
                0 - Only warnings / errors
                1 - Info messages (default)
                2 - Verbose
                3 - Debug messages
            user_opts: Options to Neurodamus overriding the simulation config file
        """
        enable_reports = not user_opts.pop("disable_reports", False)
        if logging_level is not None:
            GlobalConfig.verbosity = logging_level

        Node.__init__(self, config_file, user_opts)
        # Use the run_conf dict to avoid passing it around
        self._run_conf["EnableReports"] = enable_reports
        self._run_conf["AutoInit"] = auto_init

        if SimConfig.dry_run:
            if self._is_ngv_run:
                raise Exception("Dry run not available for ngv circuit")
            self.load_targets()
            self.create_cells()
            self.create_synapses()
            return

        if SimConfig.restore_coreneuron:
            self._coreneuron_restore()
        elif SimConfig.build_model:
            self._instantiate_simulation()

        # Remove .SUCCESS file if exists
        self._success_file = SimConfig.config_file + ".SUCCESS"
        self._remove_file(self._success_file)

    # -
    def _build_single_model(self):
        """Construct the model for a single cycle.

        This process includes:
        - Computing load balance across ranks.
        - Building the circuit by creating cells and applying configurations.
        - Establishing synaptic connections.
        - Enabling replay mechanisms if applicable.
        - Initializing the simulation if 'AutoInit' is enabled.
        """
        log_stage("================ CALCULATING LOAD BALANCE ================")
        load_bal = self.compute_load_balance()
        print_mem_usage()

        log_stage("==================== BUILDING CIRCUIT ====================")
        self.create_cells(load_bal)
        print_mem_usage()

        # Create connections
        self.create_synapses()
        print_mem_usage()

        log_stage("================ INSTANTIATING SIMULATION ================")
        # Apply replay
        self.enable_replay()
        print_mem_usage()

        if self._run_conf["AutoInit"]:
            self.init()

    # -
    def init(self):
        """Explicitly initialize, allowing users to make last changes before simulation"""
        if self._sim_ready:
            logging.warning("Simulation already initialized. Skip second init")
            return

        log_stage("Creating connections in the simulator")
        base_seed = self._run_conf.get("BaseSeed", 0)  # base seed for synapse RNG
        for syn_manager in self._circuits.all_synapse_managers():
            syn_manager.finalize(base_seed)
        print_mem_usage()

        self.enable_stimulus()
        print_mem_usage()
        self.enable_modifications()

        if self._run_conf["EnableReports"]:
            self.enable_reports()
        print_mem_usage()

        self.sim_init()
        assert self._sim_ready, "sim_init should have set this"

    @staticmethod
    def _merge_filesdat(ncycles):
        log_stage("Generating merged CoreNeuron files.dat")
        coreneuron_datadir = CoreConfig.datadir
        cn_entries = []
        for i in range(ncycles):
            log_verbose(f"files_{i}.dat")
            filename = ospath.join(coreneuron_datadir, f"files_{i}.dat")
            with open(filename, encoding="utf-8") as fd:
                first_line = fd.readline()
                nlines = int(fd.readline())
                for _ in range(nlines):
                    line = fd.readline()
                    cn_entries.append(line)

        cnfilename = ospath.join(coreneuron_datadir, "files.dat")
        with open(cnfilename, "w", encoding="utf-8") as cnfile:
            cnfile.write(first_line)
            cnfile.write(str(len(cn_entries)) + "\n")
            cnfile.writelines(cn_entries)

        logging.info(" => %s files merged successfully", ncycles)

    def _coreneuron_restore(self):
        """Restore CoreNEURON simulation state.

        This method sets up the CoreNEURON environment for restoring a simulation:
        - load targets
        - enable replay
        - enable reports (this writes also report.conf)
        - write sim.conf
        - set and link coreneuron_datadir to the old restore one
        """
        log_stage(" =============== CORENEURON RESTORE ===============")
        self.load_targets()
        self.enable_replay()
        if self._run_conf["EnableReports"]:
            self.enable_reports()

        self._coreneuron_write_sim_config(corenrn_restore=True)
        self._setup_coreneuron_datadir_from_restore()

        self._sim_ready = True

    @run_only_rank0
    def _setup_coreneuron_datadir_from_restore(self):
        """Configure the environment for restoring CoreNEURON.

        This involves:
        - setting the coreneuron_datadir
        - writing the sim.conf
        - linking the old coreneuron_datadir to the new one
        (in save_path or output_root)
        """
        self._coreneuron_configure_datadir(
            corenrn_restore=True, coreneuron_direct_mode=SimConfig.coreneuron_direct_mode
        )

        # handle coreneuron_input movements
        src_datadir = Path(SimConfig.coreneuron_datadir_restore_path())
        dst_datadir = Path(SimConfig.coreneuron_datadir_path())
        # Check if source directory exists
        if not src_datadir.exists():
            raise FileNotFoundError(
                f"Coreneuron input directory in `{src_datadir}` does not exist!"
            )

        # If the source exists,
        # remove the destination directory or symlink (if it exists)
        if dst_datadir.exists():
            if dst_datadir.is_symlink():
                # Remove the symlink
                dst_datadir.unlink()
            else:
                # Remove the folder if it's not a symlink
                shutil.rmtree(dst_datadir)

        dst_datadir.symlink_to(src_datadir)

    def compute_n_cycles(self):
        """Determine the number of model-building cycles

        It is based on configuration and system constraints.
        """
        n_cycles = SimConfig.modelbuilding_steps
        # No multi-cycle. Trivial result, this is always possible
        if n_cycles == 1:
            return n_cycles

        target = self._target_manager.get_target(self._target_spec)
        target_name = self._target_spec.name
        max_cell_count = target.max_gid_count_per_population()
        logging.info(
            "Simulation target: %s, Max cell count per population: %d", target_name, max_cell_count
        )

        if SimConfig.use_coreneuron and max_cell_count / n_cycles < MPI.size and max_cell_count > 0:
            # coreneuron with no. ranks >> no. cells
            # need to assign fake gids to artificial cells in empty threads
            # during module building fake gids start from max_gid + 1
            # currently not support engine plugin where target is loaded later
            # We can always have only 1 cycle. coreneuron throws an error if a
            # rank does not have cells during a cycle. There is a way to prevent
            # this for unbalanced multi-populations but if more than one cycle
            # happens on a rank without instantiating cells another error raises.
            # Thus, the number of cycles should be rounded down; on the safe side
            max_num_cycles = int(max_cell_count / MPI.size) or 1
            if n_cycles > max_num_cycles:
                logging.warning(
                    "Your simulation is using multi-cycle without enough cells.\n"
                    "  => Number of cycles has been automatically set to the max: %d",
                    max_num_cycles,
                )
                n_cycles = max_num_cycles
        return n_cycles

    def _build_model(self):
        """Build the model

        Internally it calls _build_single_model, over multiple
        cycles if necessary.

        Note: only relevant for coreNeuron
        """
        self._n_cycles = self.compute_n_cycles()

        # Without multi-cycle, it's a trivial model build.
        # sub_targets is False
        if self._n_cycles == 1:
            self._build_single_model()
            return

        logging.info("MULTI-CYCLE RUN: %s Cycles", self._n_cycles)
        target = self._target_manager.get_target(self._target_spec)
        TimerManager.archive(archive_name="Before Cycle Loop")

        PopulationNodes.freeze_offsets()

        if SimConfig.loadbal_mode != LoadBalanceMode.Memory:
            sub_targets = target.generate_subtargets(self._n_cycles)

        for cycle_i in range(self._n_cycles):
            logging.info("")
            logging.info("-" * 60)
            log_stage(f"==> CYCLE {cycle_i + 1} (OUT OF {self._n_cycles})")
            logging.info("-" * 60)

            self.clear_model()

            if SimConfig.loadbal_mode != LoadBalanceMode.Memory:
                for cur_target in sub_targets[cycle_i]:
                    self._target_manager.register_target(cur_target)
                    pop = next(iter(cur_target.population_names))
                    for circuit in self._sonata_circuits.values():
                        tmp_target_spec = TargetSpec(circuit.CircuitTarget)
                        if tmp_target_spec.population == pop:
                            tmp_target_spec.name = cur_target.name
                            circuit.CircuitTarget = str(tmp_target_spec)

            self._cycle_i = cycle_i
            self._build_single_model()

            # Move generated files aside (to be merged later)
            if MPI.rank == 0:
                base_filesdat = ospath.join(CoreConfig.datadir, "files")
                os.rename(base_filesdat + ".dat", base_filesdat + f"_{cycle_i}.dat")
            # Archive timers for this cycle
            TimerManager.archive(archive_name=f"Cycle Run {cycle_i + 1:d}")

        if MPI.rank == 0:
            self._merge_filesdat(self._n_cycles)

    # -
    def _instantiate_simulation(self):
        """Initialize the simulation

        - load targets
        - check connections
        - build the model
        """
        # Keep the initial RSS for the SHM file transfer calculations
        self._initial_rss = SHMUtil.get_node_rss()
        print_mem_usage()

        self.load_targets()

        # Check connection block configuration and raise warnings for overriding
        # parameters
        SimConfig.check_connections_configure(self._target_manager)

        self._build_model()

    # -
    @timeit(name="finished Run")
    def run(self, cleanup=True):
        """Prepares and launches the simulation according to the loaded config.
        If '--only-build-model' option is set, simulation is skipped.

        Args:
            cleanup (bool): Free up the model and intermediate files [default: true]
                Rationale is: the high-level run() method it's typically for a
                one shot simulation so we should cleanup. If not it can be set to False
        """
        if SimConfig.dry_run:
            log_stage("============= DRY RUN (SKIP SIMULATION) =============")
            self._dry_run_stats.display_total()
            self._dry_run_stats.display_node_suggestions()
            ranks = self._dry_run_stats.get_num_target_ranks(SimConfig.num_target_ranks)
            self._dry_run_stats.collect_all_mpi()
            try:
                self._dry_run_stats.distribute_cells_with_validation(
                    ranks, SimConfig.modelbuilding_steps
                )
            except RuntimeError:
                logging.exception("Dry run failed")
            return

        if not SimConfig.simulate_model:
            self.sim_init()
            log_stage("======== [SKIPPED] SIMULATION (MODEL BUILD ONLY) ========")
        elif not SimConfig.build_model:
            log_stage("============= SIMULATION (SKIP MODEL BUILD) =============")
            # coreneuron needs the report file created
            self._run_coreneuron()
        else:
            log_stage("======================= SIMULATION =======================")
            self.run_all()

        # Create SUCCESS file if the simulation finishes successfully
        self._touch_file(self._success_file)
        logging.info("Finished! Creating .SUCCESS file: '%s'", self._success_file)

        # Save seclamp holding currents for gap junction user corrections
        if (
            gj_target_pop := SimConfig.beta_features.get("gapjunction_target_population")
        ) and SimConfig.beta_features.get("procedure_type") == "find_holding_current":
            gj_manager = self._circuits.get_edge_manager(
                gj_target_pop, gj_target_pop, GapJunctionManager
            )
            gj_manager.save_seclamp()

        self.move_dumpcellstates_to_output_root()

        if cleanup:
            self.cleanup()

    @staticmethod
    @run_only_rank0
    def _remove_file(file_name):
        import contextlib

        with contextlib.suppress(FileNotFoundError):
            os.remove(file_name)

    @staticmethod
    @run_only_rank0
    def _touch_file(file_name):
        with open(file_name, "a", encoding="utf-8"):
            os.utime(file_name, None)
