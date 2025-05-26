"""Main module for handling and instantiating synaptical connections"""

import logging
import pickle  # noqa: S403
from pathlib import Path

import numpy as np

from .connection_manager import ConnectionManagerBase
from .core import MPI, NeuronWrapper as Nd
from .core.configuration import ConfigurationError, SimConfig
from .gap_junction_user_corrections import load_user_modifications
from .io.sonata_config import ConnectionTypes
from .io.synapse_reader import SonataReader, SynapseParameters


class GapJunctionConnParameters(SynapseParameters):
    """Glial-glial gap junction connection parameters.

    This class overrides the `_fields` attribute from the base `SynapseParameters`
    class to define a specific set of parameters relevant for gap junctions.

    The `_optional` and `_reserved` dictionaries are inherited from the base class
    and apply unchanged.

    Note:
        - Only the `_fields` dictionary is overridden.
        - The dtype construction and utility methods are reused as-is.
    """

    # Attribute names of synapse parameters, consistent with the normal synapses
    _fields = {
        "sgid": np.int64,
        "isec": np.int64,
        "offset": np.float64,
        "weight": np.float64,
        "efferent_junction_id": np.int64,
        "afferent_junction_id": np.int64,
        "ipt": np.float64,
        "location": np.float64,
    }


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
        self.seclamp_current_per_gid_recorder = None

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
            if self.seclamp_per_gid:
                # Record seclamp currents for saving to a file at the end
                self.seclamp_current_per_gid_recorder = {}
                for gid, seclamp in self.seclamp_per_gid.items():
                    self.seclamp_current_per_gid_recorder[gid] = Nd.h.Vector()
                    self.seclamp_current_per_gid_recorder[gid].record(seclamp._ref_i)

    @staticmethod
    def _finalize_conns(_final_tgid, conns, *_, **_kw):
        for conn in reversed(conns):
            conn.finalize_gap_junctions()
        return len(conns)

    def save_seclamp(self):
        """Save seclamps to a file"""
        if self.seclamp_current_per_gid_recorder:
            logging.info("Save SEClamp currents for gap junction user corrections")
            vals = {
                gid: hoc_vec.as_numpy()
                for gid, hoc_vec in self.seclamp_current_per_gid_recorder.items()
            }
            output_dir = Path(SimConfig.output_root) / "gap_junction_seclamps"
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / f"data_for_host_{MPI.rank}.p", "wb") as f:
                pickle.dump(vals, f)
