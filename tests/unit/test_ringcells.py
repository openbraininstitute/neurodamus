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

    with NamedTemporaryFile("w", suffix='.json', dir=str(tmp_path), delete=False) as config_file:
        json.dump(ringtest_baseconfig, config_file)

    yield config_file
    os.unlink(config_file.name)


@pytest.mark.parametrize("extra_config", [{"network": "$CIRCUIT_DIR/circuit_config_RingB.json",
                                            "node_set": "RingB"}])
def test_dump_RingB_2cells(simconfig_update):
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
        _check_cell(cell)
        selection = edges.afferent_edges(tgid - 1)
        _check_synapses(cell.synlist, edges, selection)
        nclist = Nd.cvode.netconlist(n._pc.gid2cell(sgid), cell, "")
        _check_netcons(sgid, nclist, edges, selection)


@pytest.mark.parametrize("extra_config", [{"node_set": "Mosaic"}])
def test_dump_RingA_RingB(simconfig_update):
    from neurodamus import Neurodamus
    from neurodamus.core import NeurodamusCore as Nd
    n = Neurodamus(simconfig_update.name, disable_reports=True)
    from neurodamus.utils.dump_cellstate import dump_cellstate
    target_rawgids = [("RingA", 1), ("RingB", 1)]
    for pop, raw_gid in target_rawgids:
        pop_offset = n.circuits.get_node_manager(pop).local_nodes.offset
        gid = raw_gid + pop_offset
        outputfile = "cellstate_" + str(gid) + ".json"
        dump_cellstate(n._pc, Nd.cvode, gid, outputfile)
        reference = REF_DIR / outputfile
        compare_json_files(outputfile, str(reference))


def compare_json_files(res_file, ref_file):
    assert os.path.isfile(res_file)
    assert os.path.isfile(ref_file)
    import json
    with open(res_file) as f_res:
        result = json.load(f_res)
    with open(ref_file) as f_ref:
        reference = json.load(f_ref)
    assert result  == reference


def _check_cell(cell):
    # check cell state from NEURON context
    assert cell.nSecAll == 3
    assert cell.x == cell.y == cell.z == 0


def _check_synapses(synlist, edges, selection):
    # check synapse state from NEURON w.r.t libsonata reader
    assert synlist.count() == selection.flat_size
    for syn in synlist:
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
    # check netcons
    assert netconlist.count() == 1
    nc = netconlist.o(0)
    assert nc.srcgid() == ref_srcgid
    assert nc.weight[0] == edges.get_attribute("conductance", selection)[0]
    assert np.isclose(nc.delay, edges.get_attribute("delay", selection)[0], rtol=1e-2)
    assert nc.threshold == SimConfig.spike_threshold
    assert nc.x == SimConfig.v_init
