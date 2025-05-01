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
        }
    ],
    indirect=True,
)
def test_neuromodulation(create_tmp_simulation_config_file):
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
    # 2 from the synapse objects from the neuron connections
    # 1 from the replay via the neuromodulatory project
    nclist = Nrn.cvode.netconlist("", cell, "")
    assert len(nclist) == 3
    assert nclist[0].srcgid() == 1
    assert nclist[1].srcgid() == 1002
    assert nclist[2].srcgid() < 0
    replay_netcon = nclist[2]
    assert replay_netcon == nclist[2]
    assert replay_netcon.syn() == nclist[1].syn()
    assert replay_netcon.weight[4] == 10
