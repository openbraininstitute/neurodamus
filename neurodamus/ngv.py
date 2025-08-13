"""Module which defines and handles Glia Cells and connectivity"""

import logging
from pathlib import Path

import libsonata
import numpy as np

from .cell_distributor import CellDistributor
from .connection import Connection
from .connection_manager import ConnectionManagerBase
from .core import (
    MPI,
    EngineBase,
    NeuronWrapper as Nd,
    mpi_no_errors,
)
from .core.configuration import ConfigurationError, GlobalConfig, LogLevel
from .io.sonata_config import ConnectionTypes
from .io.synapse_reader import SonataReader, SynapseParameters
from .metype import BaseCell
from .morphio_wrapper import MorphIOWrapper
from .utils.pyutils import append_recarray


class Astrocyte(BaseCell):
    __slots__ = ("_gluts", "glut_soma", "is_resized_sections_warning", "section_names")

    def __init__(self, gid, meinfos, circuit_conf):
        """Initialize an Astrocyte cell:

        - create the cell from Cell.hoc
        - add the morphology
        - resize sections if necessary
        - add cadifus, GlutReceive (they will be connected later)
        - record section_names for creating connections later
        """
        super().__init__()

        # Create the cell
        self._cellref = Nd.Cell(gid)  # cell instantiated with Cell.hoc
        # load and apply morphology
        morph = MorphIOWrapper(
            Path(circuit_conf.MorphologyPath)
            / f"{meinfos.morph_name}.{circuit_conf.MorphologyType}"
        )
        self._cellref.AddHocMorph(morph.morph_as_hoc())
        # Recalculate number of segments and sections
        self._cellref.geom_nseg_fixed()
        self._cellref.geom_nsec()

        logging.debug("Instantiating NGV cell gid=%d", gid)

        # assigned later
        self._gluts = {}
        self.is_resized_sections_warning = False
        for sec in self.all:
            self._init_basic_section(sec)

        # add GlutReceiveSoma (only for metabolism)
        soma = self._cellref.soma[0]
        self.glut_soma = Nd.GlutReceiveSoma(soma(0.5), sec=soma)

        self.gid = gid

    def _init_basic_section(self, sec):
        """Initialize a basic NEURON section with standard mechanisms.

        This function ensures the section has a single compartment (`nseg = 1`),
        resizing it if necessary. After, it inserts the 'cadifus' mechanism used
        for calcium diffusion. It returns whether the section had more
        than one segment before resizing.

        Parameters:
            sec (neuron.h.Section): The section to initialize.
        """
        # resize if necessary
        if sec.nseg > 1:
            self.is_resized_sections_warning = True
            sec.nseg = 1
        # add cadifus mechanism for calcium diffusion
        sec.insert("cadifus")

    def _init_endfoot_section(
        self,
        sec,
        parent_id: int,
        length: float,
        diameter: float,
        R0pas: float,  # noqa: N803
    ) -> bool:
        """Initialize an endfoot NEURON section with custom geometry and mechanisms.

        Parameters:
            sec (neuron.h.Section): The endfoot section to initialize.
            parent_id (int): Index to identify the parent section.
            length (float): Length of the endfoot section.
            diameter (float): Diameter of the endfoot section.
            R0pas (float): Passive resistance parameter for the vascouplingB mechanism.
        """
        self._init_basic_section(sec)
        sec.L = length
        sec.diam = diameter
        sec.insert("vascouplingB")
        sec(0.5).vascouplingB.R0pas = R0pas
        # connect to parent sec
        parent_sec = self.get_sec(parent_id + 1)
        sec.connect(parent_sec)

    def get_glut(self, section_id):
        """Return cached GlutReceive object for a section, creating it if needed."""
        if section_id in self._gluts:
            return self._gluts[section_id]
        sec = self.get_sec(section_id)
        glut = Nd.GlutReceive(sec(0.5), sec=sec)
        sec(0.5).cadifus._ref_glu2 = glut._ref_glut
        self._gluts[section_id] = glut
        return glut

    @property
    def gid(self) -> int:
        """Get the gid as an integer."""
        return int(self._cellref.gid)

    @gid.setter
    def gid(self, val: int):
        """Set the gid value."""
        self._cellref.gid = val

    @property
    def all(self):
        """Return the main SectionList (`_cellref.all`).

        Note:
            This list does **not** include all sections. Specifically,
            endfeet sections are excluded and can be accessed via the `endfeet` attribute.
        """
        return self._cellref.all

    @property
    def endfeet(self):
        """Returns _cellref.endfeet (SectionList)."""
        if hasattr(self._cellref, "endfeet") and self._cellref.endfeet is not None:
            return self._cellref.endfeet
        return Nd.SectionList()

    def add_endfeet(self, parent_ids, lengths, diameters, R0passes):  # noqa: N803
        assert len(parent_ids) == len(lengths) == len(diameters) == len(R0passes)
        self._cellref.execute_commands(
            [
                f"create endfoot[{len(parent_ids)}]",
                "endfeet = new SectionList()",
                'forsec "endfoot" endfeet.append',
            ]
        )
        for sec, parent_id, length, diameter, R0pas in zip(
            self.endfeet, parent_ids, lengths, diameters, R0passes
        ):
            self._init_endfoot_section(sec, parent_id, length, diameter, R0pas)

    @property
    def glut_list(self) -> list:
        # necessary for legacy compatibility with metabolism
        return [*self._gluts.values(), self.glut_soma]

    def connect2target(self, target_pp=None):
        return Nd.NetCon(self._cellref.soma[0](1)._ref_v, target_pp, sec=self._cellref.soma[0])

    @staticmethod
    def getThreshold():
        return 0.114648


class AstrocyteManager(CellDistributor):
    """Manages Astrocyte cells, extending CellDistributor

    Behaves like CellDistributor but uses the Astrocyte cell type.
    The difference lies only in the Cell Type.
    """

    CellType = Astrocyte
    _sonata_with_extra_attrs = False

    def _emit_resized_section_warnings(self):
        """Collect and emit warnings for cells that had sections resized."""
        gids = [cell.gid for cell in self._gid2cell.values() if cell.is_resized_sections_warning]
        # Gather all warning maps to rank 0
        gids = MPI.py_gather(gids, 0)

        if MPI.rank == 0:
            # flatten the list of gids
            gids = [gid for sublist in gids for gid in sublist]
            gids = sorted(gids)
            logging.warning(
                "Cells %s emitted warning: '%s'",
                sorted(gids),
                "Multi-compartment sections are not allowed at the moment."
                "Resizing to 1 compartment.",
            )

    @mpi_no_errors
    def _instantiate_cells(self, cell_type=None, **_opts):
        super()._instantiate_cells(cell_type=cell_type, **_opts)
        self._emit_resized_section_warnings()

    @mpi_no_errors
    def _instantiate_cells_dry(self, cell_type, skip_metypes, **_opts):
        super()._instantiate_cells_dry(cell_type=cell_type, skip_metypes=skip_metypes, **_opts)
        self._emit_resized_section_warnings()


class NeuroGliaConnParameters(SynapseParameters):
    """Neuron-to-glia connection parameters.

    This class overrides the `_fields` attribute from `SynapseParameters` to define
    parameters specific to neuro-glial interactions.

    The `_optional` and `_reserved` dictionaries are inherited unchanged from the base class.

    Note:
        - Only `_fields` is overridden.
        - All methods and behavior are reused from the base class.
    """

    _fields = {
        "tgid": np.int64,
        "synapse_id": np.int64,
        "astrocyte_section_id": np.int64,
        "astrocyte_segment_id": np.int64,
        "astrocyte_segment_offset": np.float64,
    }


class NeuroGlialSynapseReader(SonataReader):
    LOOKUP_BY_TARGET_IDS = False
    Parameters = NeuroGliaConnParameters
    custom_parameters = set()


class NeuroGlialConnection(Connection):
    neurons_not_found = set()
    neurons_attached = set()
    netcon_delay = 0.05
    syn_gid_offset = 1_000_000  # Below 1M is reserved for cell ids

    def add_synapse(self, syn_tpoints, params_obj, syn_id=None):
        # Only store params. Glia have mechanisms pre-created
        self._synapse_params = append_recarray(self._synapse_params, params_obj)

    def finalize(self, astrocyte, base_seed, *, base_connections=None, **kw):
        """Bind each glia connection to synapses in connections target cells via
        the assigned unique gid.
        """
        self._netcons = []
        pc = Nd.pc

        if GlobalConfig.debug_conn:
            if GlobalConfig.debug_conn == [self.tgid]:
                logging.debug("Finalizing conn %s. N params: %d", self, len(self._synapse_params))
            elif GlobalConfig.debug_conn == [self.sgid, self.tgid]:
                logging.debug("Finalizing conn %s. Params:\n%s", self, self._synapse_params)

        for syn_params in self._synapse_params:
            syn_gid = self.syn_gid_offset + syn_params.synapse_id

            # netcon to GlutReceive
            glut_idx = int(syn_params.astrocyte_section_id)
            glut_obj = astrocyte.get_glut(glut_idx)
            netcon = pc.gid_connect(syn_gid, glut_obj)
            netcon.delay = self.netcon_delay
            self._netcons.append(netcon)

            # netcon to GlutReceiveSoma for metabolism
            logging.debug("[NGV] Conn %s linking synapse id %d to Astrocyte", self, syn_gid)
            netcon = pc.gid_connect(syn_gid, astrocyte.glut_soma)
            netcon.delay = self.netcon_delay
            self._netcons.append(netcon)

        return len(self._synapse_params)


class NeuroGliaConnManager(ConnectionManagerBase):
    """A Connection Manager for Neuro-Glia connections

    NOTE: We assume the only kind of connections for Glia are Neuron-Glia
    If one day Astrocytes have connections among themselves a sub ConnectionManager
    must be used
    """

    ustate_netcon_threshold = 0.0
    ustate_netcon_delay = 0.0
    ustate_netcon_weight = 1.1

    CONNECTIONS_TYPE = ConnectionTypes.NeuroGlial
    conn_factory = NeuroGlialConnection
    SynapseReader = NeuroGlialSynapseReader

    def __init__(self, circuit_conf, target_manager, cell_manager, src_cell_manager=None, **kw):
        kw.pop("load_offsets")
        super().__init__(circuit_conf, target_manager, cell_manager, src_cell_manager, **kw)

    @staticmethod
    def _add_synapses(cur_conn, syns_params, _syn_type_restrict=None, _base_id=0):
        for syn_params in syns_params:
            cur_conn.add_synapse(None, syn_params)

    def finalize(self, base_seed=0, *_):
        """Instantiate connections to the simulator.

        This is a two-step process:
        First we create netcons to listen events on target synapses.Ustate,
        and assign them a virtual gid.
        Second, as part of NeuroGlialConnection.finalize(), we attach netcons to
        the target glia cell, listening for "signals" from the virtual gids.
        """
        logging.info("Creating virtual cells on target Neurons for coupling to GLIA...")
        base_manager = next(self._src_cell_manager.connection_managers.values())

        total_created = self._create_synapse_ustate_endpoints(base_manager)

        logging.info("(RANK 0) Created %d Virtual GIDs for synapses.", total_created)

        super().finalize(
            base_seed,
            base_connections=None,
            conn_type="NeuronGlia connections",
        )

        if NeuroGlialConnection.neurons_not_found:
            logging.warning(
                "Missing cells to couple Glia to: %d", len(NeuroGlialConnection.neurons_not_found)
            )

    @staticmethod
    def _create_synapse_ustate_endpoints(base_manager):
        """Creating an endpoint netcon to listen for events in synapse.Ustate
        Netcon ids are directly the synapse id (hence we are limited in number space)

        Note: we assume that the source synapse has a Ustate variable
        """
        pc = Nd.pc
        syn_gid_base = NeuroGlialConnection.syn_gid_offset
        total_created = 0

        for conn in base_manager.all_connections():
            syn_objs = conn.synapses
            tgid_syn_offset = syn_gid_base + conn.synapses_offset
            logging.debug("Tgid: %d, Base syn offset: %d", conn.tgid, tgid_syn_offset)

            for param_i, sec in conn.sections_with_synapses:
                if conn.synapse_params[param_i].synType < 100:  # Skip Inhibitory
                    continue
                synapse_gid = tgid_syn_offset + param_i
                pc.set_gid2node(synapse_gid, MPI.rank)
                netcon = Nd.NetCon(
                    syn_objs[param_i]._ref_Ustate,
                    None,
                    NeuroGliaConnManager.ustate_netcon_threshold,
                    NeuroGliaConnManager.ustate_netcon_delay,
                    NeuroGliaConnManager.ustate_netcon_weight,
                    sec=sec,
                )
                # set the v-gid (that is actually an synapse id) to the netcon. Useful for
                # reporting and debugging
                pc.cell(synapse_gid, netcon)
                if GlobalConfig.verbosity >= LogLevel.DEBUG:
                    netcon.record(
                        lambda tgid=conn.tgid, synapse_gid=synapse_gid: print(  # noqa: T201
                            f"[gid={tgid}] Ustate netcon event. Spiking via v-gid={synapse_gid}"
                        )
                    )

                conn._netcons.append(netcon)
                total_created += 1

        return total_created


class GlioVascularManager(ConnectionManagerBase):
    CONNECTIONS_TYPE = ConnectionTypes.GlioVascular
    InnerConnectivityCls = None  # No synapses

    def __init__(self, circuit_conf, target_manager, cell_manager, src_cell_manager=None, **kw):
        if cell_manager.circuit_target is None:
            raise ConfigurationError("Circuit target is required for GlioVascular projections")
        if "Path" not in circuit_conf:
            raise ConfigurationError("Missing GlioVascular Sonata file via 'Path' configuration")
        if "VasculaturePath" not in circuit_conf:
            raise ConfigurationError(
                "Missing Vasculature Sonata file via 'VasculaturePath' configuration"
            )

        super().__init__(circuit_conf, target_manager, cell_manager, src_cell_manager, **kw)
        self._astro_ids = self._cell_manager.local_nodes.raw_gids()
        self._gid_offset = self._cell_manager.local_nodes.offset

    def open_edge_location(self, sonata_source, circuit_conf, **__):
        logging.info("GlioVascular sonata file %s", sonata_source)
        # sonata files can have multiple populations. In building we only use one
        # per file, hence this two lines below to access the first and only pop in
        # the file
        edge_file, *pop = sonata_source.split(":")
        storage = libsonata.EdgeStorage(edge_file)
        pop_name = pop[0] if pop else next(iter(storage.population_names))
        self._gliovascular = storage.open_population(pop_name)

        storage = libsonata.NodeStorage(circuit_conf["VasculaturePath"])
        pop_name = next(iter(storage.population_names))
        self._vasculature = storage.open_population(pop_name)

    def create_connections(self, *_, **__):
        # it also creates endfeet
        logging.info("Creating GlioVascular virtual connections")
        # Retrieve endfeet selections for GLIA gids on the current processor

        for astro_id in self._astro_ids:
            self._connect_endfeet(astro_id)

    def _connect_endfeet(self, astro_id):
        endfeet = self._gliovascular.afferent_edges(astro_id - 1)  # 0-based for libsonata API
        if endfeet.flat_size > 0:
            # Get endfeet input

            parent_section_ids = self._gliovascular.get_attribute("astrocyte_section_id", endfeet)
            lengths = self._gliovascular.get_attribute("endfoot_compartment_length", endfeet)
            diameters = self._gliovascular.get_attribute("endfoot_compartment_diameter", endfeet)

            # Retrieve instantiated astrocyte
            astrocyte = self._cell_manager.gid2cell[astro_id + self._gid_offset]

            # Retrieve R0pas
            vasc_node_ids = libsonata.Selection(self._gliovascular.source_nodes(endfeet))
            d_vessel_starts = self._vasculature.get_attribute("start_diameter", vasc_node_ids)
            d_vessel_ends = self._vasculature.get_attribute("end_diameter", vasc_node_ids)
            R0passes = (d_vessel_starts + d_vessel_ends) / 4

            astrocyte.add_endfeet(parent_section_ids, lengths, diameters, R0passes)

    def finalize(self, *_, **__):
        pass  # No synpases/netcons


class NGVEngine(EngineBase):
    CellManagerCls = AstrocyteManager
    ConnectionTypes = {
        ConnectionTypes.NeuroGlial: NeuroGliaConnManager,
        ConnectionTypes.GlioVascular: GlioVascularManager,
    }
