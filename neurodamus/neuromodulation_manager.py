import logging
from itertools import chain

from .connection import Connection, NetConType, ReplayMode
from .connection_manager import SynapseRuleManager
from .core.configuration import GlobalConfig
from .io.sonata_config import ConnectionTypes
from .io.synapse_reader import SonataReader, SynapseParameters
from .utils.logging import log_all


class NeuroModulationConnection(Connection):
    __slots__ = ("_neuromod_dtc", "_neuromod_strength")

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
        self._neuromod_strength = None
        self._neuromod_dtc = None
        super().__init__(
            sgid,
            tgid,
            src_pop_id,
            dst_pop_id,
            weight_factor,
            minis_spont_rate,
            configuration,
            mod_override,
            **kwargs,
        )

    neuromod_strength = property(
        lambda self: self._neuromod_strength,
        lambda self, val: setattr(self, "_neuromod_strength", val),
    )
    neuromod_dtc = property(
        lambda self: self._neuromod_dtc, lambda self, val: setattr(self, "_neuromod_dtc", val)
    )

    def finalize(
        self,
        cell,
        base_seed=0,
        *,
        replay_mode=ReplayMode.AS_REQUIRED,
        base_managers=None,
        **_kwargs,
    ):
        """Override the finalize process from the base class.
        NeuroModulatory events do not create synapses but link to existing cell synapses.
        A neuromodulatory connection from projections with match to the closest cell synapse.
        A spike coming from the neuromodulatory event (SynapseReplay) will trigger the
        NET_RECEIVE of the existing synapse, with the weight (binary 1/0), neuromod_strength,
        neuromod_dtc, and nc_type NC_MODULATOR
        """
        logging.debug("Finalize neuromodulation connection")

        self._netcons = []
        # Initialize member lists
        self._init_artificial_stims(cell, replay_mode)

        for syn_i, sec in self.sections_with_synapses:
            syn_params = self._synapse_params[syn_i]
            # We need to get all connections since we dont know the sgid
            # TODO: improve this by extracting all the relative distances only once
            base_conns = chain.from_iterable(
                base_manager.get_connections(self.tgid) for base_manager in base_managers
            )
            syn_obj = self._find_closest_cell_synapse(syn_params, base_conns)
            if syn_obj is None:
                logging.warning("No cell synapse associated to the neuromodulatory event")
                return 0
            if self._replay is not None:
                nc = self._replay.create_on(self, sec, syn_obj, syn_params)
                nc.weight[0] = int(self.weight_factor > 0)  # weight is binary 1/0, default 1
                nc.weight[1] = self.neuromod_strength or syn_params.neuromod_strength
                nc.weight[2] = self.neuromod_dtc or syn_params.neuromod_dtc
                self.netcon_set_type(nc, syn_obj, NetConType.NC_NEUROMODULATOR)
                if GlobalConfig.debug_conn == [self.tgid]:
                    log_all(
                        logging.DEBUG,
                        "Neuromodulatory event on tgid: %d, weights: [%f, %f, %f], nc_type: %d",
                        self.tgid,
                        nc.weight[0],
                        nc.weight[1],
                        nc.weight[2],
                        NetConType.NC_NEUROMODULATOR,
                    )

        return 1

    @staticmethod
    def _find_closest_cell_synapse(syn_params, base_conns):
        """Find the closest cell synapse by the location parameter"""
        if not base_conns:
            return None
        section_i = syn_params.isec
        location_i = syn_params.location
        min_diff = 0.05
        syn_obj = None
        for base_conn in base_conns:
            for syn_j, _ in base_conn.sections_with_synapses:
                params_j = base_conn._synapse_params[syn_j]
                if params_j.isec != section_i:
                    continue
                diff = abs(params_j.location - location_i)
                if diff < min_diff:
                    syn_obj = base_conn._synapses[syn_j]
                    min_diff = diff
        return syn_obj


class ModulationConnParameters(SynapseParameters):
    """Modulatory connection parameters.

    This class defines parameters for neuromodulatory connections by overriding
    `_fields` from `SynapseParameters`.

    Notes:
        - The field names are consistent with standard synapses for compatibility.
        - `location` is computed using the HOC function `TargetManager.locationToPoint`
          with `isec`, `offset`, and `ipt`.
        - `ipt` is set to -1 (not read from data) to ensure `locationToPoint` sets
          `location = offset`.
        - `weight` is used as a placeholder for replay stimulation; it defaults to 1.0
          and is later overwritten by the actual connection weight.

    The `_optional` and `_reserved` dictionaries are inherited from the base class.
    """

    _fields = {
        "sgid": "int64",
        "delay": "float64",
        "isec": "int64",
        "offset": "float64",
        "neuromod_strength": "float64",
        "neuromod_dtc": "float64",
        "ipt": "float64",
        "location": "float64",
        "weight": "float64",
    }


class NeuroModulationSynapseReader(SonataReader):
    Parameters = ModulationConnParameters
    custom_parameters = {"isec", "ipt", "offset", "weight"}

    def _load_params_custom(self, _populate, _read):
        super()._load_params_custom(_populate, _read)
        _populate("weight", 1)


class NeuroModulationManager(SynapseRuleManager):
    CONNECTIONS_TYPE = ConnectionTypes.NeuroModulation
    conn_factory = NeuroModulationConnection
    SynapseReader = NeuroModulationSynapseReader

    def _finalize_conns(self, tgid, conns, base_seed, **kwargs):
        """Override the function from the base class.
        Retrieve the base synapse connections with the same tgid.
        Pass the base connection managers (from all src populations except the neuromodulatory one)
        to the finalize process of superclass, to be processed by NeuroModulationConnection.
        """
        base_managers = [
            manager
            for src_pop, manager in self.cell_manager.connection_managers.items()
            if src_pop != self.src_node_population
        ]
        return super()._finalize_conns(
            tgid, conns, base_seed, base_managers=base_managers, **kwargs
        )
