
from pathlib import Path

import numpy as np
import pytest
from libsonata import EdgeStorage

from neurodamus.core.coreneuron_configuration import CoreConfig
from neurodamus.core.configuration import SimConfig
from tests import utils

from ..conftest import RINGTEST_DIR


def check_cell(cell):
    """check cell state from NEURON context"""
    #assert cell.nSecAll == 3
    #assert cell.x == cell.y == cell.z == 0


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "network": str(RINGTEST_DIR / "circuit_config_RingB.json"),
            "node_set": "RingB",
            "target_simulator": "NEURON"
        }
    },
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "network": str(RINGTEST_DIR / "circuit_config_RingB.json"),
            "node_set": "RingB",
            "target_simulator": "CORENEURON"
        }
    }
], indirect=True)
def test_dump_RingB_2cells(create_tmp_simulation_config_file):
    from neurodamus import Neurodamus
    from neurodamus.core import NeuronWrapper as Nd

    n = Neurodamus(create_tmp_simulation_config_file, disable_reports=True)
    edges_file, edge_pop = SimConfig.sonata_circuits["RingB"].nrnPath.split(":")
    edge_storage = EdgeStorage(edges_file)
    edges = edge_storage.open_population(edge_pop)
    src_gids = [1, 2]
    target_gids = np.roll(src_gids, -1)
    for sgid, tgid in zip(src_gids, target_gids):
        cell = n._pc.gid2cell(tgid)
        check_cell(cell)
        selection = edges.afferent_edges(tgid - 1)

        assert cell.synlist.count() == selection.flat_size
        for syn in cell.synlist:
            utils.check_synapse(syn, edges, selection)

        nclist = Nd.cvode.netconlist(n._pc.gid2cell(sgid), cell, "")
        utils.check_netcons(sgid, nclist, edges, selection)

    if SimConfig.use_coreneuron:
        utils.check_directory(CoreConfig.datadir)


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "target_simulator": "NEURON",
            "node_set": "Mosaic"
        }
    },
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "target_simulator": "CORENEURON",
            "node_set": "Mosaic"
        }
    }
], indirect=True)
def test_dump_RingA_RingB(create_tmp_simulation_config_file):
    from neurodamus import Neurodamus
    from neurodamus.core import NeuronWrapper as Nd

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
        reference = RINGTEST_DIR / "reference" / outputfile
        utils.compare_json_files(Path(outputfile), reference)

        cell = n._pc.gid2cell(tgid)
        check_cell(cell)

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
        utils.check_netcons(sgid, nclist, edges, selection)
        utils.check_synapses(nclist, edges, selection)

    if SimConfig.use_coreneuron:
        utils.check_directory(CoreConfig.datadir)


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "target_simulator": "CORENEURON",
            "node_set": "Mosaic"
        }
    }
], indirect=True)
def test_coreneuron(create_tmp_simulation_config_file):
    from neurodamus import Neurodamus

    n = Neurodamus(create_tmp_simulation_config_file, disable_reports=True,
                   coreneuron_direct_mode=True, keep_build=True)
    n.run()
    coreneuron_data = Path(CoreConfig.datadir)
    assert coreneuron_data.is_dir() and not any(coreneuron_data.iterdir()), (
        f"{coreneuron_data} should be empty."
    )
