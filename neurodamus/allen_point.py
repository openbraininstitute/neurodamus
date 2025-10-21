from .core import (
    EngineBase,
    NeuronWrapper as Nd,
)
from .cell_distributor import CellManagerBase, CellDistributor
from .connection_manager import SynapseRuleManager, ConnectionSet
from .io.sonata_config import ConnectionTypes
from .connection import Connection
from .metype import Cell_V6
import numpy as np
import logging
from .core.configuration import SimConfig


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

class PointConnection(Connection):
    def add_synapses(self, target_manager, synapses_params, base_id=0):
        """Adds synapses in bulk.

        Args:
         - synapses_params: A SynapseParameters array (possibly view) for this conn synapses
         - base_id: The synapse base id, usually absolute offset

        """
        n_synapses = len(synapses_params)
        self._synapse_params = synapses_params
        assert n_synapses <=1, "Allen PointConnection supports max. one synapse per connection"

    def finalize(
        self,
        cell,
        base_seed=0,
        **kwargs):
        """When all parameters are set, create synapses and netcons

        Args:
            cell: The cell to create synapses and netcons on.
            base_seed: base seed value (Default: None - no adjustment)
            replay_mode: Policy to initialize replay in this conection

        """
        # Initialize member lists
        # self._synapses = compat.List()  # Used by ConnUtils
        self._netcons = []
        # self._init_artificial_stims(cell, replay_mode)
        n_syns = 0
        for syn_params in self.synapse_params:
            n_syns += 1
            nc = Nd.pc.gid_connect(self.sgid, cell.CellRef.pointcell)
            nc.delay = self.syndelay_override or syn_params.delay
            nc.weight[0] = syn_params.weight * self.weight_factor
            nc.threshold = SimConfig.spike_threshold
            self._netcons.append(nc)
        #TODO: replay 

        return n_syns



class AllenPointConnectionManager(SynapseRuleManager):
    conn_factory = PointConnection


class AllenPointEngine(EngineBase):
    CellManagerCls = AllenPointNeuronManager
    InnerConnectivityCls = AllenPointConnectionManager
    ConnectionTypes = {
        ConnectionTypes.PointNeuron: AllenPointConnectionManager,
    }
    CircuitPrecedence = 0