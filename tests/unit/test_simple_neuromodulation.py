"""Test neuromodulations using the toy ringtest data as such:
neuromodulatory projection: virtual_neurons->RingB, 0->0, 1->0
The post synatic cell RingB gid 0 has 2 synapses:
    RingA->RingB : 0->0
    RingB->RingB : 1->0
The neuromodulatory spikes are injected via replay from virtual neuron gid 1
Therefore, the neuromodulator applies netcon to the synapse from connection RingB->RingB : 1->0
"""

import numpy as np
import pytest

from ..conftest import RINGTEST_DIR
from neurodamus import Neurodamus
from neurodamus.neuromodulation_manager import NeuroModulationManager


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

    gid = 1001
    edges_BB = nd.circuits.get_edge_manager("RingB", "RingB")
    assert [conn.sgid for conn in edges_BB.get_connections(gid)] == [1002]
    edges_AB = nd.circuits.get_edge_manager("RingA", "RingB")
    assert [conn.sgid for conn in edges_AB.get_connections(gid)] == [1]
    edges_VB = nd.circuits.get_edge_manager("virtual_neurons", "RingB")
    assert isinstance(edges_VB, NeuroModulationManager)
    assert [conn.sgid for conn in edges_VB.get_connections(gid)] == [2001, 2002]

    cell = nd._pc.gid2cell(gid)
    # check cell gid 1001 has 2 syns from neurons,
    # neuromodulatory projection don't create addition synapses
    assert cell.synlist.count() == 2
    # check netcons targeting cell gid 1001, it should have 3 netcons,
    # 2 from the synapse objects from the neuron connections (B->B, A->B)
    # 1 from the replay via the neuromodulatory project (virtual_neurons->B)
    nclist = Nrn.cvode.netconlist("", cell, "")
    assert len(nclist) == 3
    assert nclist[0].srcgid() == 1
    assert nclist[1].srcgid() == 1002
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

    cell = nd._pc.gid2cell(1001)
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
