import json
import os
import pytest
import numpy as np
from pathlib import Path
from tempfile import NamedTemporaryFile
from neurodamus.core.configuration import SimConfig
from libsonata import EdgeStorage


SIM_DIR = Path(__file__).parent.parent.absolute() / "simulations" / "ringtest"
REF_DIR = SIM_DIR / "reference"
CONFIG_FILE = str(SIM_DIR / "simulation_config.json")


@pytest.fixture
def simconfig_update(ringtest_baseconfig, extra_config, tmp_path):
    ringtest_baseconfig.update(extra_config)

    with NamedTemporaryFile("w", suffix=".json", dir=str(tmp_path), delete=False) as config_file:
        json.dump(ringtest_baseconfig, config_file)

    yield config_file
    os.unlink(config_file.name)


@pytest.mark.parametrize(
    "extra_config",
    [
        {
            "network": "$CIRCUIT_DIR/circuit_config_RingB.json",
            "target_simulator": "NEURON",
            "node_set": "RingB",
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
            ],
        }
    ],
)
def test_synweight_delay_neuron(simconfig_update):
    from neurodamus import Neurodamus
    from neurodamus.core import NeurodamusCore as Nd

    n = Neurodamus(simconfig_update.name, disable_reports=True)
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
    # check netcons
    assert netconlist.count() == selection.flat_size
    for idx, nc in enumerate(netconlist):
        nc = netconlist.o(idx)
        assert nc.weight[0] == edges.get_attribute("conductance", selection)[idx] * weight_factor
