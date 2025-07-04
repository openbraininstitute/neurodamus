"""Implementation of the core Connection classes"""

import contextlib
import logging
import re
from enum import Enum

import numpy as np

from .core import NeuronWrapper as Nd
from .core.configuration import ConfigurationError, GlobalConfig, SimConfig
from .utils import compat
from .utils.logging import log_all
from .utils.pyutils import append_recarray


class ReplayMode(Enum):
    """Replay instantiation mode."""

    NONE = 0
    """Instantiate no replay NetCons"""

    AS_REQUIRED = 1
    """Instantiate Replay netcons as required for this run.
    Subsequent Save-Restore may not work"""

    COMPLETE = 2
    """Instantiate Replay Netcons on all Connections so that
    users may add arbitrary new replays in Restore phases"""


class NetConType(Enum):
    """NetCon Type"""

    NC_PRESYN = 0
    NC_SPONTMINI = 1
    NC_REPLAY = 2
    NC_NEUROMODULATOR = 10

    def __int__(self):
        return self.value


class ConnectionBase:
    """The Base implementation for cell connections identified by src-dst gids"""

    __slots__ = (
        "_delay_vec",
        "_delayweight_vec",
        "_dst_pop_id",
        "_netcons",
        "_src_pop_id",
        "_syn_offset",
        "_synapse_params",
        "_synapses",
        "locked",
        "sgid",
        "syndelay_override",
        "tgid",
        "weight_factor",
    )

    def __init__(
        self,
        sgid,
        tgid,
        src_pop_id=0,
        dst_pop_id=0,
        weight_factor=1,
        syndelay_override=None,
        synapses_offset=0,
    ):
        """Initializes a base connection object

        Args:
            sgid: presynaptic gid
            tgid: postsynaptic gid
            src_pop_id: The id of the source node population
            dst_pop_id: The id of the target node population
            weight_factor: the weight factor to be applied to the connection. Default: 1
            syndelay_override: The delay for this connection, overriding "delay" property
            synapses_offset: The offset within the edge file (metadata)
        """
        self.sgid = int(sgid or -1)
        self.tgid = int(tgid)
        self.weight_factor = weight_factor
        self.syndelay_override = syndelay_override
        self.locked = False
        self._syn_offset = synapses_offset
        self._src_pop_id = src_pop_id
        self._dst_pop_id = dst_pop_id
        self._synapse_params = None
        # Initialized in specific routines
        self._netcons = None
        self._synapses = ()
        self._delay_vec = Nd.Vector()
        self._delayweight_vec = Nd.Vector()

    synapse_params = property(lambda self: self._synapse_params)
    synapses = property(lambda self: self._synapses)
    synapses_offset = property(lambda self: self._syn_offset)
    population_id = property(lambda self: (self._src_pop_id, self._dst_pop_id))

    # Subclasses must implement instantiation of their connections in the simulator
    def finalize(self, cell, base_seed=0, *args, **kw):
        raise NotImplementedError("finalize must be implemented in sub-class")

    # Parameters Live update / Configuration
    # --------------------------------------
    def update_conductance(self, new_g):
        """Updates all synapses conductance"""
        for syn in self._synapses:
            syn.g = new_g

    def update_weights(self, weight):
        """Change the weights of the netcons generated when connecting
        the source and target gids represented in this connection
        """
        for nc in self._netcons:
            nc.weight[0] = weight

    def add_delayed_weight(self, delay, weight):
        """Update the delayed connection vectors in the synapse.

        Args:
           delay: the delay time for the new weight
           weight: the weight to adjust to at this time
        """
        self._delay_vec.append(delay)
        self._delayweight_vec.append(weight)

    # Recent synapses accept a signal type (synapse, replay, etc...)
    # The index at which we set it (in `.weight` array) is a top-level Neuron var
    # However checking/getting it is slow. So we cache globally
    _netcon_signal_type_index_cache = {}

    _match_index = re.compile(r"\[[0-9]+\]")
    """Regex to match indexes of hoc objects, useful to get mechs name"""

    @classmethod
    def netcon_set_type(cls, netcon, syn_obj, nc_type):
        """Find nc_type_param from the synapse global variable and set via the netcon weight"""
        nc_param_name = cls._match_index.sub("", f"nc_type_param_{syn_obj}")
        nc_param_index = cls._netcon_signal_type_index_cache.get(nc_param_name)
        if nc_param_index is None:  # False -> not supported by this model
            nc_param_index = getattr(Nd, nc_param_name, False)
            if nc_param_index:
                nc_param_index = int(nc_param_index)

            cls._netcon_signal_type_index_cache[nc_param_name] = nc_param_index

        if nc_param_index:
            netcon.weight[nc_param_index] = int(nc_type)

    def __str__(self):
        return f"[{self.sgid}->{self.tgid}]"


class Connection(ConnectionBase):
    """A Connection object serves as a container for synapses formed from
    a presynaptic and a postsynaptic gid, including Points where those
    synapses are placed (stored in TPointList)
    """

    __slots__ = (
        "_configurations",
        "_mod_override",
        "_replay",
        "_spont_minis",
        "_synapse_ids",
        "_synapse_points_x",
        "_synapse_sections",
        "minis_spont_rate",
    )

    _AMPANMDA_Helper = None
    _GABAAB_Helper = None
    ConnUtils = None  # Collection of hoc routines to speedup execution
    _mod_overrides = set()

    @classmethod
    def _init_hmod(cls):
        if cls._AMPANMDA_Helper is not None:
            return
        h = Nd.require("AMPANMDAHelper", "GABAABHelper")
        cls._AMPANMDA_Helper = h.AMPANMDAHelper
        cls._GABAAB_Helper = h.GABAABHelper
        cls.ConnUtils = h.ConnectionUtils()
        cls._pc = Nd.pc

    def __init__(
        self,
        sgid,
        tgid,
        src_pop_id=0,
        dst_pop_id=0,
        weight_factor=1.0,
        minis_spont_rate=None,
        configuration=None,
        mod_override=None,
        **kwargs,
    ):
        """Creates a connection object

        Args:
            sgid: presynaptic gid
            tgid: postsynaptic gid
            weight_factor: the weight factor to be applied to the connection. Default: 1
            configuration: Any synapse configurations that should be applied
                when the synapses are instantiated (or None)
            minis_spont_rate: rate for spontaneous minis. Default: None
            mod_override: Alternative Synapse type. Default: None (use standard Inh/Exc)
        """
        self._init_hmod()
        super().__init__(sgid, tgid, src_pop_id, dst_pop_id, weight_factor, **kwargs)
        self.minis_spont_rate = minis_spont_rate
        self._mod_override = mod_override
        self._synapse_sections = []
        self._synapse_points_x = compat.array("d")
        self._synapse_ids = compat.array("i")  # replaced by np.array for bulk add syn
        self._configurations = [configuration] if configuration is not None else []
        # Artificial stimulus sources
        self._spont_minis = None
        self._replay = None

    def add_synapse_configuration(self, configuration):
        """Add a synapse configuration command to the list.
        All commands are executed on synapse creation
        """
        if configuration is not None:
            self._configurations.append(configuration)

    @property
    def mod_override(self):
        """_mod_override getter"""
        return self._mod_override

    @mod_override.setter
    def mod_override(self, mod_override):
        """Set a valid mod override"""
        assert mod_override is not None, "ModOverride cannot be None"
        self._mod_override = mod_override

    @property
    def sections_with_synapses(self):
        """Generator over all sections containing synapses, yielding pairs
        (section_index, section)
        """
        for syn_i, sc in enumerate(self._synapse_sections):
            # All locations, on and off node should be in this list, but
            # only synapses/netcons on-node should be returned
            if not sc.exists():
                continue
            yield syn_i, sc.sec

    def add_synapses(self, target_manager, synapses_params, base_id=0):
        """Adds synapses in bulk.

        Args:
         - synapses_params: A SynapseParameters array (possibly view) for this conn synapses
         - base_id: The synapse base id, usually absolute offset

        """
        n_synapses = len(synapses_params)
        synapse_ids = np.arange(base_id, base_id + n_synapses, dtype="uint64")
        # We may need to skip invalid synapses (e.g. on Axon)
        mask = np.full(n_synapses, fill_value=True)
        for i, syn_params in enumerate(synapses_params):
            syn_point = target_manager.location_to_point(
                self.tgid, syn_params["isec"], syn_params["ipt"], syn_params["offset"]
            )
            syn_params["location"] = syn_point.x[0]
            section = syn_point.sclst[0]

            if section is None or not section.exists():
                target_point_str = (
                    f"({syn_params.isec:.0f} {syn_params.ipt:.0f} {syn_params.offset:.4f})"
                )
                logging.warning(
                    "SKIPPED Synapse %s on gid %d. Src gid: %d. Deleted TPoint %s",
                    base_id + i,
                    self.tgid,
                    self.sgid,
                    target_point_str,
                )
                mask[i] = False
                continue

            # These are normal lists/arrays, so we cant use masks
            self._synapse_sections.append(section)
            self._synapse_points_x.append(syn_point.x[0])

        if not mask.all():
            synapses_params = synapses_params[mask]
            synapse_ids = synapse_ids[mask]

        if self._synapse_params is None or len(self._synapse_params) == 0:  # None or empty
            self._synapse_params = synapses_params
            self._synapse_ids = synapse_ids
        else:
            self._synapse_params = np.concatenate(
                (self._synapse_params, synapses_params), dtype=self._synapse_params.dtype
            )
            self._synapse_ids = np.concatenate((self._synapse_ids, synapse_ids))

    def add_synapse(self, syn_tpoints, params_obj, syn_id=None):
        """Adds a synapse in given location to this Connection.
        NOTE: This procedure can have a significant impact when called multiple
        times. Consider add_synapses to add multiple synapses in bulk

        Args:
            syn_tpoints: TPointList with one point on the tgid where the
                associated synapse exists
            params_obj: Parameters object for the Synapse to be placed
            syn_id: Optional id for the synapse to be used for seeding rng
        """
        # Update four lists:
        # - synapse_sections
        # - synapse_points_x
        # - synapse_params (slow!)
        # - synapse_ids

        for i, sc in enumerate(syn_tpoints.sclst):
            self._synapse_sections.append(sc)
            self._synapse_points_x.append(syn_tpoints.x.x[i])

        self._synapse_params = append_recarray(self._synapse_params, params_obj)
        params_obj.location = syn_tpoints.x[0]  # helper

        if syn_id is None:
            syn_id = len(self._synapse_sections)
        self._synapse_ids.append(syn_id)

    def add_single(self, cell_manager, syn_params, syn_id):
        """Add synapse config in target cell"""
        soma = Nd.SectionRef(sec=cell_manager.get_cell(self.tgid).soma[0])
        self._synapse_ids.append(syn_id)
        self._synapse_sections.append(soma)
        self._synapse_points_x.append(0.5)
        if self._synapse_params is None:
            self._synapse_params = [syn_params]
        else:
            self._synapse_params.append(syn_params)

    def replay(self, tvec, start_delay=0.0):
        """The synapses connecting these gids are to be activated using
        predetermined timings.

        Args:
            tvec: time for spike events from the sgid
            start_delay: When the events may start to be delivered
        """
        assert self._netcons is None, "Replay must be setup prior to finalize()"
        hoc_tvec = Nd.Vector(tvec[tvec >= start_delay])
        logging.debug("Replaying %d spikes on %d - %d", hoc_tvec.size(), self.sgid, self.tgid)
        logging.debug(" > First replay event for connection at %f", hoc_tvec.x[0])

        if self._replay is None:
            self._replay = ReplayStim()
        self._replay.add_spikes(hoc_tvec)
        return len(self._replay)

    def finalize(
        self,
        cell,
        base_seed=0,
        *,
        replay_mode=ReplayMode.AS_REQUIRED,
        attach_src_cell=True,
    ):
        """When all parameters are set, create synapses and netcons

        Args:
            cell: The cell to create synapses and netcons on.
            base_seed: base seed value (Default: None - no adjustment)
            replay_mode: Policy to initialize replay in this conection

        """
        # Initialize member lists
        self._synapses = compat.List()  # Used by ConnUtils
        self._netcons = []
        self._init_artificial_stims(cell, replay_mode)
        n_syns = 0
        for syn_i, sec in self.sections_with_synapses:
            x = self._synapse_points_x[syn_i]
            syn_params = self._synapse_params[syn_i]

            with Nd.section_in_stack(sec):
                syn_obj = self._create_synapse(
                    cell, syn_params, x, self._synapse_ids[syn_i], base_seed
                )
                n_syns += 1

            self._synapses.append(syn_obj)
            # syn_obj.verboseLevel = self.tgid  # debugging purposes

            if attach_src_cell:
                self._attach_source_cell(syn_obj, syn_params)

            if self._spont_minis is not None:
                self._spont_minis.create_on(self, sec, x, syn_obj, syn_params, base_seed)

            if self._replay is not None:
                self._replay.create_on(self, sec, syn_obj, syn_params)

            # Delayed connections
            if self._delay_vec is not None:
                if not hasattr(syn_obj, "setup_delay_vecs"):
                    logging.error(
                        "py-neurodamus no longer supports delayed connections in the legacy "
                        "format, which among others was incompatible with CoreNeuron. "
                        "Please consider updating your models with the lastest synapse"
                        "implementation (from models/common) or use py-neurodamus <= 1.3.1"
                    )
                    raise ValueError(f"{syn_obj} does not support delayed connections")

                syn_obj.setup_delay_vecs(self._delay_vec, self._delayweight_vec)

        self._configure_synapses()
        return n_syns

    def _init_artificial_stims(self, cell, replay_mode=ReplayMode.AS_REQUIRED):
        shall_create_replay = replay_mode == ReplayMode.COMPLETE or (
            replay_mode == ReplayMode.AS_REQUIRED and self._replay and self._replay.has_data()
        )

        # Release objects if not needed
        if not shall_create_replay:
            self._replay = None
        # if spont_minis not set by user, set with default rates from circuit if available
        if self.minis_spont_rate is None:
            if cell.inh_mini_frequency or cell.exc_mini_frequency:
                self._spont_minis = InhExcSpontMinis(
                    cell.inh_mini_frequency, cell.exc_mini_frequency
                )
        else:
            self._spont_minis = SpontMinis(self.minis_spont_rate)
        # Release spont_minis object if it evaluates to false (rates are 0)
        if not self._spont_minis:
            self._spont_minis = None

        # Delayed vecs: release if not used, sort if over 1 value
        if self._delay_vec is None or self._delayweight_vec is None:
            # they should be set together
            assert self._delayweight_vec == self._delay_vec
            return
        total_delays = self._delay_vec.size()
        if total_delays == 0:
            self._delay_vec = None
            self._delayweight_vec = None
        elif total_delays > 1:
            sort_indx = self._delay_vec.sortindex()
            self._delay_vec = Nd.Vector(total_delays).index(self._delay_vec, sort_indx)
            self._delayweight_vec = Nd.Vector(total_delays).index(self._delayweight_vec, sort_indx)

    def _attach_source_cell(self, syn_obj, syn_params):
        # See `neurodamus-core.Connection` for explanation. Also pc.gid_connect
        nc = self._pc.gid_connect(self.sgid, syn_obj)
        self.netcon_set_type(nc, syn_obj, NetConType.NC_PRESYN)
        nc.delay = self.syndelay_override or syn_params.delay
        nc.weight[0] = syn_params.weight * self.weight_factor
        nc.threshold = SimConfig.spike_threshold
        self._netcons.append(nc)
        return nc

    def _create_synapse(self, cell, params_obj, x, syn_id, base_seed):
        """Instantiate synapses (GABBAB inhibitory, AMPANMDA excitatory, etc)
        passing the creation helper the synapse params.

        Created synapses are appended to the corresponding cell lists.
        Third-party Synapse types are supported via the synapse-override
        configuration.

        Args:
            cell: The cell object
            params_obj: SynapseParameters object for the synapse to be
                placed at a single location.
            x: distance into the currently accessed section (cas)
            syn_id: Synapse id (NRN: determined by row number)
            base_seed: base seed to adjust synapse RNG - added to
                MCellRan4's low index parameter

        """
        is_inh = params_obj["synType"] < 100
        if self._mod_override is not None:
            self._mod_overrides.add(self._mod_override)
            override_helper = self._mod_override + "Helper"
            helper_cls = getattr(Nd.h, override_helper)
            add_params = (self._src_pop_id, self._dst_pop_id)
        else:
            helper_cls = self._GABAAB_Helper if is_inh else self._AMPANMDA_Helper
            add_params = (self._src_pop_id, self._dst_pop_id)

        syn_helper = helper_cls(self.tgid, params_obj, x, syn_id, base_seed, *add_params)

        # set the synapse conductance obtained from the synapse file
        # this variable is exclusively used for delay connections
        if hasattr(syn_helper.synapse, "conductance"):
            syn_helper.synapse.conductance = params_obj["weight"]

        # set the default value of synapse NMDA_ratio/GABAB_ratio from circuit
        conductance_ratio = float(params_obj["conductance_ratio"])
        if conductance_ratio >= 0.0 and self._mod_override is None:
            self._update_conductance_ratio(syn_helper.synapse, is_inh, conductance_ratio)

        cell.CellRef.synHelperList.append(syn_helper)
        cell.CellRef.synlist.append(syn_helper.synapse)
        return syn_helper.synapse

    def finalize_gap_junctions(self):
        """When all parameters are set, create synapses and netcons

        Args:
            cell: The cell to create synapses and netcons on.
        """
        self._synapses = compat.List()
        self._netcons = []

        for syn_i, sec in self.sections_with_synapses:
            x = self._synapse_points_x[syn_i]
            active_params = self._synapse_params[syn_i]
            gap_junction = Nd.Gap(x, sec=sec)

            dbg_conn = GlobalConfig.debug_conn
            if dbg_conn and dbg_conn in ([self.tgid], [self.sgid, self.tgid]):
                log_all(
                    logging.DEBUG,
                    (
                        "connect %f to %f [efferent_junction_id: %f], "
                        "[afferent_junction_id: %f] (weight: %f)"
                    ),
                    self.sgid,
                    self.tgid,
                    active_params.efferent_junction_id,
                    active_params.afferent_junction_id,
                    active_params.weight,
                )

            with Nd.section_in_stack(sec):
                self._pc.target_var(
                    gap_junction,
                    gap_junction._ref_vgap,
                    active_params.efferent_junction_id,
                )
                self._pc.source_var(sec(x)._ref_v, active_params.afferent_junction_id)
            gap_junction.g = active_params.weight
            self._synapses.append(gap_junction)

    # ------------------------------------------------------------------
    # Parameters Live update / Configuration
    # ------------------------------------------------------------------

    def update_weights(self, weight, update_also_replay_netcons=False):
        """Change the weights of the existing netcons

        Args:
            weight: The new weight
            update_also_replay_netcons: Whether weights shall be applied to
                replay netcons as well
        """
        super().update_weights(weight)
        if update_also_replay_netcons and self._replay is not None:
            for nc in self._replay.netcons:
                nc.weight[0] = weight

    def _update_conductance_ratio(self, syn_obj, is_inhibitory, value):
        """Update the relevant conductance ratio of synapse object
        inhibitory synapse : GABAB_ratio
        excitatory synapse : NMDA_ratio
        """
        dbg_conn = GlobalConfig.debug_conn
        if dbg_conn and dbg_conn in ([self.tgid], [self.sgid, self.tgid]):
            log_all(
                logging.DEBUG,
                "[%d->%d] Update synapse %s ratio to %.6f",
                self.sgid,
                self.tgid,
                "GABAB" if is_inhibitory else "NMDA",
                value,
            )
        if is_inhibitory:
            syn_obj.GABAB_ratio = value
        else:
            syn_obj.NMDA_ratio = value

    def _configure(self, synapses, configuration):
        res = self.ConnUtils.executeConfigure(synapses, configuration)
        if res > 0:
            raise ConfigurationError(f"Errors found in configuration: {configuration}")

    def _configure_synapses(self):
        """Internal helper to apply all the configuration statements to
        the created synapses
        """
        for config in self._configurations:
            self.configure_synapses(config)

    def configure_synapses(self, configuration):
        """Helper function to execute a configuration statement (hoc)
        on all connection synapses.
        """
        self._configure(self._synapses, configuration)

    def restart_events(self):
        """Restart the artificial events, coming from Replay or Spont-Minis"""
        if self._spont_minis is not None:
            self._spont_minis.restart_events()
        if self._replay is not None:
            self._replay.restart_events()

    def __del__(self):
        """Clear Random123 objects when connection is deleted"""
        for syn in self._synapses:
            with contextlib.suppress(AttributeError):
                syn.clearRNG()


class ArtificialStim:
    """Base class for artificial Stims, namely Replay and Minis"""

    __slots__ = ("netcons", "netstims")

    _bbss = None
    """SaveState object. Initialized on first use"""

    def __init__(self):
        self.netstims = []
        self.netcons = []
        if ArtificialStim._bbss is None:
            ArtificialStim._bbss = Nd.BBSaveState()

    def _store(self, netstim, netcon):
        if netstim is not None:
            self._bbss.ignore(netstim)
            self.netstims.append(netstim)
        self.netcons.append(netcon)

    def restart_events(self):
        for stim in self.netstims:
            stim.restartEvent()


class SpontMinis(ArtificialStim):
    """A class creating/holding spont minis of a connection"""

    __slots__ = ("_rng_info", "rate_vec")

    tbins_vec = None
    """Neurodamus uses a constant rate, so tbin is always containing only 0
    """  # Nd.Vector must be called later to avoid init Neuron on import

    @classmethod
    def _cls_init(cls):
        cls.tbins_vec = Nd.Vector(1)
        cls.tbins_vec.x[0] = 0.0

    def __init__(self, minis_spont_rate):
        super().__init__()
        self.tbins_vec or self._cls_init()
        self._rng_info = Nd.RNGSettings()
        self.rate_vec = None

        if minis_spont_rate is not None:  # Allow None (used by subclass)
            self.set_rate(minis_spont_rate)

    def get_rate(self):
        return self.rate_vec[0] if self.rate_vec is not None else None

    def set_rate(self, rate):
        if rate < 0:
            raise ValueError(f"Spont minis rate cannot be negative {rate:g}")

        # Check if initialized. Dont recreate in order to enable in-simulation udates
        if self.rate_vec is None:
            self.rate_vec = Nd.Vector(1)
        self.rate_vec.x[0] = rate

    rate = property(get_rate, set_rate)

    def has_data(self):
        return self.rate_vec is not None

    def create_on(self, conn, sec, position, syn_obj, syn_params, _base_seed, _rate_vec=None):
        """Inserts a SpontMini stim into the given synapse"""
        rate_vec = _rate_vec or self.rate_vec  # allow override (private API)
        if GlobalConfig.debug_conn in ([conn.tgid], [conn.sgid, conn.tgid]):
            log_all(
                logging.DEBUG,
                "Creating Spont Minis on %d-%d, Rate: %f",
                conn.sgid,
                conn.tgid,
                rate_vec[0],
            )

        ips = Nd.InhPoissonStim(position, sec=sec)
        ips.setTbins(self.tbins_vec)
        ips.setRate(rate_vec)
        # In Neuron we can limit the duration of the Minis since InhPoissonStim's are
        # recreated on restore. CoreNeuron reuses them and we dont know final duration
        if SimConfig.use_neuron:
            ips.duration = Nd.tstop

        # A simple NetCon will do, as the synapse and cell are local.
        netcon = Nd.NetCon(ips, syn_obj, sec=sec)
        netcon.delay = 0.1
        netcon.weight[0] = syn_params.weight * conn.weight_factor
        conn.netcon_set_type(netcon, syn_obj, NetConType.NC_SPONTMINI)
        self._store(ips, netcon)

        src_pop_id, dst_pop_id = conn.population_id
        rng_seed = self._rng_info.getMinisSeed()
        tgid_seed = conn.tgid + 250

        seed2 = src_pop_id * 65536 + dst_pop_id + rng_seed
        ips.setRNGs(
            syn_obj.synapseID + 200,
            tgid_seed,
            seed2 + 300,
            syn_obj.synapseID + 200,
            tgid_seed,
            seed2 + 350,
        )

    def __bool__(self):
        """Object is considered False in case rate is not positive"""
        return bool(self.get_rate())

    def __del__(self):
        for ips in self.netstims:
            ips.setRNGs()


class InhExcSpontMinis(SpontMinis):
    """Extends SpontMinis to handle two spont rates: Inhibitory & Excitatory"""

    def __init__(self, spont_rate_inh, spont_rate_exc):
        super().__init__(spont_rate_inh or None)  # positive rate, otherwise None
        self.rate_vec_exc = None
        if spont_rate_exc:
            self.rate_vec_exc = Nd.Vector(1)
            self.rate_vec_exc.x[0] = spont_rate_exc

    def create_on(self, conn, sec, position, syn_obj, syn_params, *args):
        rate_vec = self.rate_vec if syn_params.synType < 100 else self.rate_vec_exc
        if rate_vec:
            # there's a spont rate for this kind of synapse
            super().create_on(conn, sec, position, syn_obj, syn_params, *args, _rate_vec=rate_vec)

    def has_data(self):
        return self.rate_vec is not None or self.rate_vec_exc is not None

    def get_rate(self):
        return (super().get_rate(), self.rate_vec_exc[0] if self.rate_vec_exc is not None else None)

    def __bool__(self):
        """Object is considered False in case no rate is positive"""
        return any(self.get_rate())


class ReplayStim(ArtificialStim):
    """A class creating/holding replays of a connection"""

    __slots__ = ("time_vec",)

    def __init__(self):
        super().__init__()
        self.time_vec = None

    def create_on(self, conn, sec, syn_obj, syn_params):
        """Inserts a replay stim into the given synapse"""
        vecstim = None
        if self.has_data():
            vecstim = Nd.VecStim(sec=sec)
            vecstim.play(self.time_vec)

        if GlobalConfig.debug_conn in ([conn.tgid], [conn.sgid, conn.tgid]):
            log_all(
                logging.DEBUG,
                "Creating Replay on %d-%d, times: %s",
                conn.sgid,
                conn.tgid,
                self.time_vec.as_numpy() if self.has_data() else "N/A",
            )

        nc = Nd.NetCon(
            vecstim,
            syn_obj,
            10,
            conn.syndelay_override or syn_params.delay,
            syn_params.weight,
            sec=sec,
        )
        nc.weight[0] = syn_params.weight * conn.weight_factor
        conn.netcon_set_type(nc, syn_obj, NetConType.NC_REPLAY)
        self._store(vecstim, nc)
        return nc

    def add_spikes(self, hoc_tvec):
        """Appends replay spikes from a time vector to the main replay vector"""
        if self.time_vec is None:
            self.time_vec = hoc_tvec
        else:
            self.time_vec.append(hoc_tvec)
        self.time_vec.sort()

    def has_data(self):
        return self.time_vec is not None

    def __len__(self):
        return self.time_vec.size() if self.time_vec else -1
