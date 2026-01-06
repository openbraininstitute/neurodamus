import numpy as np

from .cell_distributor import CellDistributor
from .connection import Connection
from .connection_manager import SynapseRuleManager
from .core import (
    EngineBase,
    NeuronWrapper as Nd,
)
from .core.configuration import SimConfig
from .io.sonata_config import ConnectionTypes
from .io.synapse_reader import SonataReader, SynapseParameters
from .metype import Cell_V6


class AllenPointCell(Cell_V6):
    def connect2target(self, target_pp=None):
        """Connects MEtype cell to target

        Args:
            target_pp: target point process [default: None]

        Returns: NetCon obj
        """
        return Nd.NetCon(self.CellRef.pointcell, target_pp)


class AllenPointNeuronManager(CellDistributor):
    CellType = AllenPointCell


class PointProcessConnParameters(SynapseParameters):
    """Allen's point process connection parameters.

    This class defines parameters for allen's chemical connections by overriding
    `_fields` from `SynapseParameters`.

    Notes:
        - The field names are consistent with standard synapses for compatibility.
        - `location` is computed using the HOC function `TargetManager.locationToPoint`
          with `isec`, `offset`, and `ipt`.
        - `ipt` is set to -1 (not read from data) to ensure `locationToPoint` sets
          `location = offset`.
          and is later overwritten by the actual connection weight.

    The `_optional` and `_reserved` dictionaries are inherited from the base class.
    """

    _fields = {
        "sgid": np.int64,
        "delay": np.float64,
        "isec": np.int32,
        "ipt": np.int32,
        "offset": np.float64,
        "weight": np.float64,
    }


class PointProcessSynSynapseReader(SonataReader):
    Parameters = PointProcessConnParameters


class PointConnection(Connection):
    def add_synapses(self, target_manager, synapses_params, base_id=0):
        """Adds synapses in bulk.

        Args:
         - synapses_params: A SynapseParameters array (possibly view) for this conn synapses
         - base_id: The synapse base id, usually absolute offset

        """
        n_synapses = len(synapses_params)
        self._synapse_params = synapses_params
        assert n_synapses <= 1, "Allen PointConnection supports max. one synapse per connection"

    def finalize(self, cell, base_seed=None, attach_src_cell=True, **kw):
        """Create netcons for point neuron connections

        Args:
            cell: The cell to create netcons on.

        """
        self._netcons = []
        n_syns = 0
        for syn_params in self.synapse_params:
            n_syns += 1
            if attach_src_cell:
                nc = Nd.pc.gid_connect(self.sgid, cell.CellRef.pointcell)
                nc.delay = self.syndelay_override or syn_params.delay
                nc.weight[0] = syn_params.weight * self.weight_factor
                nc.threshold = SimConfig.spike_threshold
                self._netcons.append(nc)
            if self._replay is not None and self._replay.has_data():
                vecstim = Nd.VecStim()
                vecstim.play(self._replay.time_vec)
                nc = Nd.NetCon(
                    vecstim,
                    cell.CellRef.pointcell,
                    10,
                    self.syndelay_override or syn_params.delay,
                    syn_params.weight,
                )
                nc.weight[0] = syn_params.weight * self.weight_factor
                self._replay._store(vecstim, nc)

        return n_syns


class AllenPointConnectionManager(SynapseRuleManager):
    conn_factory = PointConnection
    SynapseReader = PointProcessSynSynapseReader


class AllenPointEngine(EngineBase):
    CellManagerCls = AllenPointNeuronManager
    InnerConnectivityCls = AllenPointConnectionManager
    ConnectionTypes = {
        ConnectionTypes.PointProcess: AllenPointConnectionManager,
    }
    CircuitPrecedence = 2


class Exp2SynConnParameters(SynapseParameters):
    """Exp2Syn connection parameters.

    This class defines parameters for allen's chemical connections by overriding
    `_fields` from `SynapseParameters`.

    Notes:
        - The field names are consistent with standard synapses for compatibility.
        - `location` is computed using the HOC function `TargetManager.locationToPoint`
          with `isec`, `offset`, and `ipt`.
        - `ipt` is set to -1 (not read from data) to ensure `locationToPoint` sets
          `location = offset`.

    The `_reserved` dictionaries are inherited from the base class.
    """

    _fields = {
        "sgid": np.int64,
        "delay": np.float64,
        "isec": np.int32,
        "ipt": np.int32,
        "offset": np.float64,
        "weight": np.float64,
        "synType": np.int32,
        "conductance_ratio": np.float64,
        "location": np.float64,  # needed by connection class
        "tau1": np.float64,
        "tau2": np.float64,
        "erev": np.float64,
    }

    _optional = {"conductance_ratio": -1.0, "synType": -1}  # dummy value needed by connection class


class Exp2SynSynapseReader(SonataReader):
    Parameters = Exp2SynConnParameters


class Exp2SynSynapseRuleManager(SynapseRuleManager):
    CONNECTIONS_TYPE = ConnectionTypes.Exp2Syn
    SynapseReader = Exp2SynSynapseReader
