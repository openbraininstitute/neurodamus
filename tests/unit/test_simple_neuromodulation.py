"""Test neuromodulations using the toy ringtest data as such:
neuromodulatory projection: virtual_neurons->RingB, 0->0, 1->0
The post synatic cell RingB gid 0 has 2 synapses:
    RingA->RingB : 0->0
    RingB->RingB : 1->0
The neuromodulatory spikes are injected via replay from virtual neuron gid 1
Therefore, the neuromodulator applies netcon to the synapse from connection RingB->RingB : 1->0
"""

from itertools import chain

import numpy as np
import pytest

from ..conftest import RINGTEST_DIR
from neurodamus import Neurodamus
from neurodamus.neuromodulation_manager import NeuroModulationConnection, NeuroModulationManager


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {
                "network": str(RINGTEST_DIR / "circuit_config_neuromodulation.json"),
                "node_set": "Mosaic",
                "target_simulator": "NEURON",
                "inputs": {
                    "neuromodulatory": {
                        "node_set": "RingB",
                        "input_type": "spikes",
                        "delay": 0.0,
                        "duration": 5000.0,
                        "module": "synapse_replay",
                        "spike_file": str(RINGTEST_DIR / "neuromodulation/proj_spikes.h5"),
                    }
                },
            },
        },
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {
                "network": str(RINGTEST_DIR / "circuit_config_neuromodulation.json"),
                "node_set": "Mosaic",
                "target_simulator": "CORENEURON",
                "inputs": {
                    "neuromodulatory": {
                        "node_set": "RingB",
                        "input_type": "spikes",
                        "delay": 0.0,
                        "duration": 5000.0,
                        "module": "synapse_replay",
                        "spike_file": str(RINGTEST_DIR / "neuromodulation/proj_spikes.h5"),
                    }
                },
            },
        },
    ],
    indirect=True,
)
def test_neuromodulation(create_tmp_simulation_config_file):
    """
    Test neuromodulation process in NEURON and CoreNEURON
    """
    # the import of neuron must be at function level,
    # otherwise impact other tests even done in forked process
    from neurodamus.core import NeuronWrapper as Nrn

    nd = Neurodamus(create_tmp_simulation_config_file)

    gid = 1000
    edges_BB = nd.circuits.get_edge_manager("RingB", "RingB")
    assert [conn.sgid for conn in edges_BB.get_connections(gid)] == [1001]
    edges_AB = nd.circuits.get_edge_manager("RingA", "RingB")
    assert [conn.sgid for conn in edges_AB.get_connections(gid)] == [0]
    edges_VB = nd.circuits.get_edge_manager("virtual_neurons", "RingB")
    assert isinstance(edges_VB, NeuroModulationManager)
    assert [conn.sgid for conn in edges_VB.get_connections(gid)] == [2000, 2001]

    cell = nd._pc.gid2cell(gid)
    # check cell gid 1001 has 2 syns from neurons,
    # neuromodulatory projection don't create addition synapses
    assert cell.synlist.count() == 2
    # check netcons targeting cell gid 1001, it should have 3 netcons,
    # 2 from the synapse objects from the neuron connections (B->B, A->B)
    # 1 from the replay via the neuromodulatory project (virtual_neurons->B)
    nclist = Nrn.cvode.netconlist("", cell, "")
    assert len(nclist) == 3
    assert nclist[0].srcgid() == 0
    assert nclist[1].srcgid() == 1001
    assert nclist[2].srcgid() < 0
    replay_netcon = nclist[2]
    assert replay_netcon.pre().hname() == "VecStim[0]"  # source obj is VecStim
    assert replay_netcon.syn() == nclist[1].syn()  # target syn is the same as netcon[1], src 1002
    assert replay_netcon.weight[0] == 1
    assert np.isclose(replay_netcon.weight[1], 0.2)
    assert np.isclose(replay_netcon.weight[2], 75)
    assert replay_netcon.weight[4] == 10

    nd.run()


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {
                "network": str(RINGTEST_DIR / "circuit_config_neuromodulation.json"),
                "node_set": "Mosaic",
                "target_simulator": "NEURON",
            },
        }
    ],
    indirect=True,
)
def test_find_closest_cell_synapse(create_tmp_simulation_config_file):
    nd = Neurodamus(create_tmp_simulation_config_file)
    edges_VB = nd.circuits.get_edge_manager("virtual_neurons", "RingB")
    base_managers = [
        manager
        for src_pop, manager in edges_VB.cell_manager.connection_managers.items()
        if src_pop != "virtual_neurons"
    ]
    base_conns = list(
        chain.from_iterable(base_manager.get_connections(1000) for base_manager in base_managers)
    )

    conns = list(edges_VB.get_connections(post_gids=1000))
    assert len(conns) == 2
    assert len(conns[0].synapse_params) == 1
    assert len(conns[1].synapse_params) == 2

    # found closest synapse w.r.t neuromodulation connection sec=1, location 0.22
    syn_obj_1 = NeuroModulationConnection._find_closest_cell_synapse(
        conns[0].synapse_params[0], base_conns
    )
    assert np.isclose(conns[0].synapse_params[0].location, 0.22)
    assert np.isclose(syn_obj_1.get_loc(), 0.25)

    # found closest synapse w.r.t neuromodulation connection sec=1, location 0.78
    syn_obj_2 = NeuroModulationConnection._find_closest_cell_synapse(
        conns[1].synapse_params[0], base_conns
    )
    assert np.isclose(conns[1].synapse_params[0].location, 0.78)
    assert np.isclose(syn_obj_2.get_loc(), 0.75)

    # not found, abs(0.81 - 0.75 or 0.22) > 0.05
    syn_obj_3 = NeuroModulationConnection._find_closest_cell_synapse(
        conns[1].synapse_params[1], base_conns
    )
    assert np.isclose(conns[1].synapse_params[1].location, 0.81)
    assert syn_obj_3 is None


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {
                "network": str(RINGTEST_DIR / "circuit_config_neuromodulation.json"),
                "node_set": "Mosaic",
                "target_simulator": "NEURON",
                "inputs": {
                    "neuromodulatory": {
                        "node_set": "RingB",
                        "input_type": "spikes",
                        "delay": 0.0,
                        "duration": 5000.0,
                        "module": "synapse_replay",
                        "spike_file": str(RINGTEST_DIR / "neuromodulation/proj_spikes.h5"),
                    }
                },
                "connection_overrides": [
                    {
                        "name": "neuromodulatory",
                        "source": "virtual_neurons",
                        "target": "RingB",
                        "weight": 2,
                        "neuromodulation_strength": 0.33,
                        "neuromodulation_dtc": 77.7,
                    }
                ],
            },
        }
    ],
    indirect=True,
)
def test_override_strength_dtc(create_tmp_simulation_config_file):
    """
    Test the overriding of neuromodulation_strength and neuromodulation_dtc
      via simulation config file
    """
    from neurodamus.core import NeuronWrapper as Nrn

    nd = Neurodamus(create_tmp_simulation_config_file)

    cell = nd._pc.gid2cell(1000)
    # check cell gid 1001 has 2 syns from neurons,
    # neuromodulatory projection don't create addition synapses
    assert cell.synlist.count() == 2
    nclist = Nrn.cvode.netconlist("", cell, "")
    assert len(nclist) == 3
    replay_netcon = nclist[2]
    assert replay_netcon.pre().hname() == "VecStim[0]"  # source obj is VecStim
    assert replay_netcon.syn() == nclist[1].syn()  # target syn is the same as netcon[1]
    assert replay_netcon.weight[0] == 1  # neuromodulatory weight is binary
    assert np.isclose(replay_netcon.weight[1], 0.33)
    assert np.isclose(replay_netcon.weight[2], 77.7)
    assert replay_netcon.weight[4] == 10

    nd.run()
