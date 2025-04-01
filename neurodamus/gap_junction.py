"""Main module for handling and instantiating synaptical connections"""

import logging

import numpy as np

from .connection_manager import ConnectionManagerBase
from .core.configuration import ConfigurationError, SimConfig
from .gj_user_corrections import load_user_modifications
from .io.sonata_config import ConnectionTypes
from .io.synapse_reader import SonataReader, SynapseParameters


class GapJunctionConnParameters(SynapseParameters):
    # Attribute names of synapse parameters, consistent with the normal synapses
    _synapse_fields = (
        "sgid",
        "isec",
        "offset",
        "weight",
        "efferent_junction_id",
        "afferent_junction_id",
        "ipt",
        "location",
    )

    @classmethod
    def create_array(cls, length):
        npa = np.recarray(length, cls.dtype)
        npa.location = 0.5
        return npa


class GapJunctionSynapseReader(SonataReader):
    Parameters = GapJunctionConnParameters
    parameter_mapping = {
        "weight": "conductance",
    }
    # "isec", "ipt", "offset" are custom parameters as in base class


class GapJunctionManager(ConnectionManagerBase):
    """The GapJunctionManager is similar to the SynapseRuleManager. It will
    open dedicated connectivity files which will have the locations and
    conductance strengths of gap junctions detected in the circuit.
    The user will have the capacity to scale the conductance weights.
    """

    CONNECTIONS_TYPE = ConnectionTypes.GapJunction
    SynapseReader = GapJunctionSynapseReader

    def __init__(self, gj_conf, target_manager, cell_manager, src_cell_manager=None, **kw):
        """Initialize GapJunctionManager, opening the specified GJ
        connectivity file.

        Args:
            gj_conf: The gaps junctions configuration block / dict
            target_manager: The TargetManager which will be used to query
                targets and translate locations to points
            cell_manager: The cell manager of the target population
            src_cell_manager: The cell manager of the source population
        """
        if cell_manager.circuit_target is None:
            raise ConfigurationError(
                "No circuit target. Required when initializing GapJunctionManager"
            )
        if "Path" not in gj_conf:
            raise ConfigurationError("Missing GapJunction 'Path' configuration")

        super().__init__(gj_conf, target_manager, cell_manager, src_cell_manager, **kw)
        self._src_target_filter = target_manager.get_target(
            cell_manager.circuit_target, src_cell_manager.population_name
        )
        self.holding_ic_per_gid = None
        self.seclamp_per_gid = None

    def create_connections(self, *_, **_kw):
        """Gap Junctions dont use connection blocks, connect all belonging to target"""
        self.connect_all()

    def configure_connections(self, conn_conf):
        """Gap Junctions dont configure_connections"""

    def finalize(self, *_, **_kw):
        super().finalize(conn_type="Gap-Junctions")
        if (
            gj_target_pop := SimConfig.beta_features.get("gapjunction_target_population")
        ) and self.cell_manager.population_name == gj_target_pop:
            logging.info("Load user modification on %s", self)
            self.holding_ic_per_gid, self.seclamp_per_gid = load_user_modifications(self)

    def _finalize_conns(self, _final_tgid, conns, *_, **_kw):
        for conn in reversed(conns):
            conn.finalize_gap_junctions()
        return len(conns)
