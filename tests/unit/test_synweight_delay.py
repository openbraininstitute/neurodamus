import pytest
import numpy as np
from pathlib import Path
from neurodamus.core.configuration import SimConfig
from libsonata import EdgeStorage

SIM_DIR = Path(__file__).parent.parent.absolute() / "simulations" / "ringtest"


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "network": str(SIM_DIR / "circuit_config_RingB.json"),
            "node_set": "RingB",
            "target_simulator": "NEURON",
            "connection_overrides": [
                {
                    "name": "init_conn",
                    "source": "RingB",
                    "target": "RingB",
                    "weight": 2,
                },
                {
                    "name": "delayed_conn",
                    "source": "RingB",
                    "target": "RingB",
                    "weight": 0.5,
                    "delay": 10,
                },
            ]
        }
    }
], indirect=True)
def test_synweight_delay_neuron(create_tmp_simulation_config_file):
    from neurodamus import Neurodamus
    from neurodamus.core import NeurodamusCore as Nd

    n = Neurodamus(create_tmp_simulation_config_file, disable_reports=True)
    edges_file, edge_pop = SimConfig.extra_circuits["RingB"].nrnPath.split(":")
    edge_storage = EdgeStorage(edges_file)
    edges = edge_storage.open_population(edge_pop)
    src_gids = [1, 2]
    target_gids = np.roll(src_gids, -1)
    for sgid, tgid in zip(src_gids, target_gids):
        cell = n._pc.gid2cell(tgid)
        selection = edges.afferent_edges(tgid - 1)
        nclist = Nd.cvode.netconlist(n._pc.gid2cell(sgid), cell, "")
        _check_netcon_weight(nclist, edges, selection, weight_factor=2)

    # run simulation 10s, and check the delayed weight
    n.solve(10)

    for sgid, tgid in zip(src_gids, target_gids):
        cell = n._pc.gid2cell(tgid)
        selection = edges.afferent_edges(tgid - 1)
        nclist = Nd.cvode.netconlist(n._pc.gid2cell(sgid), cell, "")
        _check_netcon_weight(nclist, edges, selection, weight_factor=0.5)


def _check_netcon_weight(netconlist, edges, selection, weight_factor):
    """Check synapse weight in netcon w.r.t the values in edges file read by libsonata"""
    assert netconlist.count() == selection.flat_size
    for idx, nc in enumerate(netconlist):
        nc = netconlist.o(idx)
        assert nc.weight[0] == edges.get_attribute("conductance", selection)[idx] * weight_factor
