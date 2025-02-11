import json
import os
import pytest
from pathlib import Path
from tempfile import NamedTemporaryFile


SIM_DIR = Path(__file__).parent.parent.absolute() / "simulations" / "ringtest"
REF_DIR = SIM_DIR / "reference"
CONFIG_FILE = str(SIM_DIR / "simulation_config.json")


@pytest.fixture
def simconfig_update(ringtest_baseconfig, extra_config):
    ringtest_baseconfig.update(extra_config)

    with NamedTemporaryFile("w", suffix='.json', delete=False) as config_file:
        json.dump(ringtest_baseconfig, config_file)

    yield config_file
    os.unlink(config_file.name)


# @pytest.mark.skip()
@pytest.mark.parametrize("extra_config", [{"network": "$CIRCUIT_DIR/circuit_config_RingB.json",
                                            "node_set": "RingB"}])
def test_dump_RingB_2cells(simconfig_update):
    from neurodamus import Neurodamus
    from neurodamus.core import NeurodamusCore as Nd
    from neurodamus.core.configuration import SimConfig
    from libsonata import EdgeStorage
    n = Neurodamus(simconfig_update.name, disable_reports=True)
    edges_file, edge_pop = n._extra_circuits["RingB"].nrnPath.split(":")
    edge_storage = EdgeStorage(edges_file)
    edges = edge_storage.open_population(edge_pop)
    test_gids = [1, 2]
    for gid in test_gids:
        cell = n._pc.gid2cell(gid)

        # check cell state from NEURON context
        assert cell.nSecAll == 3
        assert cell.x == cell.y == cell.z == 0

        # check synapse state from NEURON w.r.t libsonata reader
        selection = edges.afferent_edges(gid - 1)
        assert cell.synlist.count() == selection.flat_size
        for syn in cell.synlist:
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

        # check netcons 1->2->1
        nclist = Nd.cvode.netconlist("", cell, "")
        assert nclist.count() == 1
        nc = nclist.o(0)
        assert nc.srcgid() == (gid + 1) % 2 if gid >= 2 else gid
        assert nc.weight[0] == edges.get_attribute("conductance", selection)[0]
        assert nc.threshold == SimConfig.spike_threshold


@pytest.mark.parametrize("extra_config", [{"node_set": "Mosaic"}])
def test_dump_RingA_RingB(simconfig_update):
    from neurodamus import Neurodamus
    from neurodamus.core import NeurodamusCore as Nd
    n = Neurodamus(simconfig_update.name, disable_reports=True)
    from neurodamus.utils.dump_cellstate import dump_cellstate
    test_gids = [1, 1001]
    for gid in test_gids:
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
