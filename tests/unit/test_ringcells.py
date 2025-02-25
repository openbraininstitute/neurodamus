import json
import pytest
import numpy as np
from pathlib import Path
from neurodamus.core.configuration import SimConfig
from libsonata import EdgeStorage


SIM_DIR = Path(__file__).parent.parent.absolute() / "simulations" / "ringtest"
REF_DIR = SIM_DIR / "reference"
CONFIG_FILE = str(SIM_DIR / "simulation_config.json")


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "network": str(SIM_DIR / "circuit_config_RingB.json"),
            "node_set": "RingB"
            }
    }
], indirect=True)
def test_dump_RingB_2cells(create_tmp_simulation_config_file):
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
        _check_cell(cell)
        selection = edges.afferent_edges(tgid - 1)

        assert cell.synlist.count() == selection.flat_size
        for syn in cell.synlist:
            _check_synapse(syn, edges, selection)

        nclist = Nd.cvode.netconlist(n._pc.gid2cell(sgid), cell, "")
        _check_netcons(sgid, nclist, edges, selection)


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "node_set": "Mosaic"
            }
    }
], indirect=True)
def test_dump_RingA_RingB(create_tmp_simulation_config_file):
    from neurodamus import Neurodamus
    from neurodamus.core import NeurodamusCore as Nd

    n = Neurodamus(create_tmp_simulation_config_file, disable_reports=True)
    from neurodamus.utils.dump_cellstate import dump_cellstate

    connections = [
        [("RingA", 3), ("RingA", 1)],
        [("RingB", 2), ("RingB", 1)],
        [("RingA", 1), ("RingB", 1)],
    ]

    for (s_pop, s_rawgid), (t_pop, t_rawgid) in connections:
        tpop_offset = n.circuits.get_node_manager(t_pop).local_nodes.offset
        tgid = t_rawgid + tpop_offset
        spop_offset = n.circuits.get_node_manager(s_pop).local_nodes.offset
        sgid = s_rawgid + spop_offset

        # dump cell/synapses/netcons states to a json and compare with ref
        outputfile = "cellstate_" + str(tgid) + ".json"
        dump_cellstate(n._pc, Nd.cvode, tgid, outputfile)
        reference = REF_DIR / outputfile
        compare_json_files(Path(outputfile), reference)

        cell = n._pc.gid2cell(tgid)
        _check_cell(cell)

        if s_pop == t_pop:
            edges_file, edge_pop = \
                n.circuits.get_edge_managers(t_pop, t_pop)[0].circuit_conf.nrnPath.split(":")
        else:
            edges_file, edge_pop = \
                n.circuits.get_edge_managers(s_pop, t_pop)[0].circuit_conf["Path"].split(":")
        edge_storage = EdgeStorage(edges_file)
        edges = edge_storage.open_population(edge_pop)
        selection = edges.afferent_edges(t_rawgid - 1)

        nclist = Nd.cvode.netconlist(n._pc.gid2cell(sgid), cell, "")
        for syn in _check_netcons(sgid, nclist, edges, selection):
            _check_synapse(syn, edges, selection)


def compare_json_files(res_file: Path, ref_file: Path):
    """compare two json files"""
    assert res_file.exists()
    assert ref_file.exists()
    with open(res_file) as f_res:
        result = json.load(f_res)
    with open(ref_file) as f_ref:
        reference = json.load(f_ref)
    assert result == reference


def _check_cell(cell):
    """check cell state from NEURON context"""
    assert cell.nSecAll == 3
    assert cell.x == cell.y == cell.z == 0


def _check_synapse(syn, edges, selection):
    """check synapse state from NEURON w.r.t libsonata reader"""
    syn_id = int(syn.synapseID)
    syn_type_id = edges.get_attribute("syn_type_id", selection)[syn_id]
    if syn_type_id < 100:
        assert "ProbGABAAB_EMS" in syn.hname()
        assert syn.tau_d_GABAA == edges.get_attribute("decay_time", selection)[syn_id]
    else:
        assert "ProbAMPANMDA_EMS" in syn.hname()
        assert syn.tau_d_AMPA == edges.get_attribute("decay_time", selection)[syn_id]
    assert syn.Use == edges.get_attribute("u_syn", selection)[syn_id]
    assert syn.Dep == edges.get_attribute("depression_time", selection)[syn_id]
    assert syn.Fac == edges.get_attribute("facilitation_time", selection)[syn_id]

    if edges.get_attribute("n_rrp_vesicles", selection)[syn_id] >= 0:
        assert syn.Nrrp == edges.get_attribute("n_rrp_vesicles", selection)[syn_id]


def _check_netcons(ref_srcgid, netconlist, edges, selection):
    """check netcons and yield the associated synpase object"""
    assert netconlist.count() == selection.flat_size
    for idx, nc in enumerate(netconlist):
        nc = netconlist.o(idx)
        assert nc.srcgid() == ref_srcgid
        assert nc.weight[0] == edges.get_attribute("conductance", selection)[idx]
        assert np.isclose(nc.delay, edges.get_attribute("delay", selection)[idx], rtol=1e-2)
        assert nc.threshold == SimConfig.spike_threshold
        assert nc.x == SimConfig.v_init
        yield nc.syn()
