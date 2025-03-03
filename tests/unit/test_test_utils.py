from pathlib import Path

import pytest
from libsonata import EdgeStorage

from neurodamus.core.configuration import SimConfig
from tests import utils

SIM_DIR = Path(__file__).parent.parent.absolute() / "simulations" / "ringtest"
REF_DIR = SIM_DIR / "reference"
CONFIG_FILE = str(SIM_DIR / "simulation_config.json")


def get_edges_data(create_tmp_simulation_config_file):
    """ Convenience function to extract some basic info about the circuit """

    from neurodamus import Neurodamus
    from neurodamus.core import NeurodamusCore as Nd

    n = Neurodamus(create_tmp_simulation_config_file, disable_reports=True)
    edges_file, edge_pop = SimConfig.extra_circuits["RingB"].nrnPath.split(":")
    edge_storage = EdgeStorage(edges_file)
    edges = edge_storage.open_population(edge_pop)
    sgid, tgid = 1, 2
    cell = n._pc.gid2cell(tgid)
    selection = edges.afferent_edges(tgid - 1)
    nclist = Nd.cvode.netconlist(n._pc.gid2cell(sgid), cell, "")
    return sgid, tgid, cell, edges, selection, nclist


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "network": str(SIM_DIR / "circuit_config_RingB.json"),
            "node_set": "RingB",
            "target_simulator": "NEURON"
        }
    }
], indirect=True)
def test_check_netcon(create_tmp_simulation_config_file):
    sgid, _, _, edges, selection, nclist = get_edges_data(create_tmp_simulation_config_file)

    # base
    utils.check_netcon(sgid, 0, next(iter(nclist)), edges, selection)
    # unused value. Same behavior as before
    utils.check_netcon(sgid, 0, next(iter(nclist)), edges, selection, unused_value=0)
    # wrong weight
    with pytest.raises(AssertionError):
        utils.check_netcon(sgid, 0, next(iter(nclist)), edges, selection, weight=4321)


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "network": str(SIM_DIR / "circuit_config_RingB.json"),
            "node_set": "RingB",
            "target_simulator": "NEURON"
        }
    }
], indirect=True)
def test_check_netcons(create_tmp_simulation_config_file):
    sgid, _, _, edges, selection, nclist = get_edges_data(create_tmp_simulation_config_file)
    # base
    utils.check_netcons(sgid, nclist, edges, selection)
    # wrong length
    with pytest.raises(AssertionError):
        utils.check_netcons(sgid, [], edges, selection)
    # wrong type
    with pytest.raises(TypeError):
        utils.check_netcons(sgid, [0], edges, selection)


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "network": str(SIM_DIR / "circuit_config_RingB.json"),
            "node_set": "RingB",
            "target_simulator": "NEURON"
        }
    }
], indirect=True)
def test_check_synapse(create_tmp_simulation_config_file):
    sgid, _, _, edges, selection, nclist = get_edges_data(create_tmp_simulation_config_file)

    # base
    utils.check_synapse(next(iter(nclist)).syn(), edges, selection)
    # unused value. Same behavior as before
    utils.check_synapse(next(iter(nclist)).syn(), edges, selection, unused_value=0)
    # wrong decay_time
    with pytest.raises(AssertionError):
        utils.check_synapse(next(iter(nclist)).syn(), edges, selection, decay_time=4321)


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "network": str(SIM_DIR / "circuit_config_RingB.json"),
            "node_set": "RingB",
            "target_simulator": "NEURON"
        }
    }
], indirect=True)
def test_check_synapses(create_tmp_simulation_config_file):
    _, _, _, edges, selection, nclist = get_edges_data(create_tmp_simulation_config_file)
    # base
    utils.check_synapses(nclist, edges, selection)
    # wrong length
    with pytest.raises(AssertionError):
        utils.check_synapses([], edges, selection)
    # wrong type
    with pytest.raises(TypeError):
        utils.check_synapses([0], edges, selection)
