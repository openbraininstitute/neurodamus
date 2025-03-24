from pathlib import Path

import pytest
import numpy as np
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


def test_merge_dicts_simple():
    parent = {"A": 1, "B": 2}
    child = {"A": 2., "C": 3}
    expected = {"A": 2, "B": 2, "C": 3}

    assert utils.merge_dicts(parent, child) == expected


def test_merge_dicts_nested():
    parent = {"A": {"x": 1., "y": 2}, "B": 3}
    child = {"A": {"x": 2, "z": 3}, "C": 4}
    expected = {"A": {"x": 2, "y": 2, "z": 3}, "B": 3, "C": 4}
    assert utils.merge_dicts(parent, child) == expected


def test_merge_dicts_with_type_mismatch():
    parent = {"A": 1}
    child = {"A": "string"}
    with pytest.raises(TypeError):
        utils.merge_dicts(parent, child)


def test_merge_dicts_with_empty_parent():
    parent = {}
    child = {"A": 1, "B": 2}
    expected = {"A": 1, "B": 2}
    assert utils.merge_dicts(parent, child) == expected


def test_merge_dicts_with_empty_child():
    parent = {"A": 1, "B": 2}
    child = {}
    expected = {"A": 1, "B": 2}
    assert utils.merge_dicts(parent, child) == expected


def test_merge_dicts_with_parent_and_child_empty():
    parent = {}
    child = {}
    expected = {}
    assert utils.merge_dicts(parent, child) == expected


def test_merge_dicts_deeply_nested():
    parent = {"A": {"B": {"C": 1}}}
    child = {"A": {"B": {"D": 2.}}}
    expected = {"A": {"B": {"C": 1, "D": 2}}}
    assert utils.merge_dicts(parent, child) == expected


def test_check_is_subset():
    dic = {'A': 1, 'B': {'C': 2., 'D': {'E': 3.}}, 'F': 4}
    subset = {'A': 1}
    utils.check_is_subset(dic, subset)
    subset = {'A': 1., 'B': {'C': 2}}
    utils.check_is_subset(dic, subset)
    subset = {'A': 1, 'B': {'D': {'E': 3}}}
    utils.check_is_subset(dic, subset)
    subset = {'A': 1, 'B': {'D': {'E': 3.0}}}
    utils.check_is_subset(dic, subset)


def test_check_is_subset_fail():
    dic = {'A': 1, 'B': {'C': 2, 'D': {'E': 3}}, 'F': 4}
    subset = {'A': 1, 'C': 2}
    with pytest.raises(AssertionError):
        utils.check_is_subset(dic, subset)
    subset = {'A': 1, 'B': {'C': 2, 'D': 3}}
    with pytest.raises(AssertionError):
        utils.check_is_subset(dic, subset)
    subset = {'A': 1, 'B': {'C': 3}}
    with pytest.raises(AssertionError):
        utils.check_is_subset(dic, subset)
    subset = {'A': 1, 'B': {'C': 2.1}}
    with pytest.raises(AssertionError):
        utils.check_is_subset(dic, subset)


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "network": str(SIM_DIR / "circuit_config_RingB.json"),
            "target_simulator": "NEURON",
            "run": {
                "tstop": 32
            }
        }
    }
], indirect=True)
def test_merge_simulation_configs(create_tmp_simulation_config_file):
    import json
    with open(create_tmp_simulation_config_file, "r") as f:
        config_data = json.load(f)
        assert config_data["run"]["tstop"] == 32
        assert np.isclose(config_data["run"]["dt"], 0.1)
        assert config_data["run"]["random_seed"] == 1122
        assert len(config_data["run"]) == 3


def test_check_signal_peaks():
    x = np.array([-60., -59., -20., -30., -50., -40., -55., -59., -10., -15., -30., -49., -60.])
    ref_pos = [2, 5, 8]
    utils.check_signal_peaks(x, ref_pos)

    x_ramp = x + np.arange(len(x)) * 2
    utils.check_signal_peaks(x_ramp, ref_pos)

    x_ramp = x + np.arange(len(x)) * -2
    utils.check_signal_peaks(x_ramp, ref_pos)
