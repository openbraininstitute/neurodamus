from pathlib import Path
from unittest.mock import Mock

import numpy as np
import numpy.testing as npt

from neurodamus.cell_distributor import CellDistributor
from neurodamus.connection_manager import ConnectionManagerBase
from neurodamus.core.nodeset import SelectionNodeSet
from neurodamus.gap_junction import GapJunctionSynapseReader
from neurodamus.io.synapse_reader import SonataReader
from neurodamus.target_manager import NodesetTarget
from neurodamus.utils.memory import DryRunStats

SIM_DIR = Path(__file__).parent.parent.absolute() / "simulations"


# def test_gapjunction_sonata_reader():
#     sonata_file = SIM_DIR / "mini_thalamus_sonata/gapjunction/edges.h5"
#     sonata_reader = GapJunctionSynapseReader(sonata_file)
#     syn_params_sonata = sonata_reader.get_synapse_parameters(0) # 0-base
#     ref_junction_id_pre = np.array([10257., 43930., 226003., 298841., 324744.,
#                                     1094745., 1167632., 1172523., 1260104.])
#     ref_junction_id_post = np.array([14., 52., 71., 76., 78., 84., 89., 90., 93.])
#     ref_weight = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
#     npt.assert_allclose(syn_params_sonata.efferent_junction_id, ref_junction_id_pre)
#     npt.assert_allclose(syn_params_sonata.afferent_junction_id, ref_junction_id_post)
#     npt.assert_allclose(syn_params_sonata.weight, ref_weight)


# def test_syn_read_counts():
#     sonata_file = SIM_DIR / "usecase3/local_edges_A.h5"
#     reader = SonataReader(sonata_file, "NodeA__NodeA__chemical")

#     full_counts = reader.get_counts(np.array([0, 1, 2], dtype=int))
#     assert len(full_counts) == 3  # dataset has only two but the count=0 must be set
#     assert full_counts[0] == 2
#     assert full_counts[1] == 2
#     assert full_counts[2] == 0

#     conn_counts = reader.get_conn_counts([0])
#     assert len(conn_counts) == 1
#     assert conn_counts[0] == {1: 2}

#     # Will reuse cache
#     conn_counts = reader.get_conn_counts([0, 1])
#     assert len(conn_counts) == 2
#     assert conn_counts[0] == {1: 2}  # [1->0] 2 synapses
#     assert conn_counts[1] == {0: 2}  # [0->1] 2 synapses

#     # Fully from cache
#     conn_counts = reader.get_conn_counts([1])
#     assert len(conn_counts) == 1
#     assert conn_counts[1] == {0: 2}  # [0->1] 2 synapses


def test_conn_manager_syn_stats():
    """Test _get_conn_stats in isolation using a mocked instance of SynapseRuleManager
    """
    sonata_file = SIM_DIR / "usecase3/local_edges_A.h5"
    cell_manager = Mock(CellDistributor)
    cell_manager.population_name = "pop-A"

    stats = DryRunStats()
    stats.pop_metype_gids = {"pop-A": {"metype-x": [0, 1], "metype-y": [2]}}
    conn_manager = ConnectionManagerBase(None, None, cell_manager, None, dry_run_stats=stats)
    conn_manager._synapse_reader = SonataReader(sonata_file, "NodeA__NodeA__chemical")

    target_ns = NodesetTarget("nodeset1", [SelectionNodeSet([0, 1])], [SelectionNodeSet([0, 1, 2, 3])])
    total_synapses_metype_x = conn_manager._get_conn_stats(target_ns)

    print(conn_manager._synapse_reader.get_conn_counts([0, 1]))


    assert total_synapses_metype_x == 2
    assert stats.metype_cell_syn_average["metype-x"] == 1

    # With a larger target we will count just the difference
    target_ns2 = NodesetTarget("nodeset2", [SelectionNodeSet([0, 1, 2, 3])], [SelectionNodeSet([0, 1, 2, 3])])
    additional_synapses = conn_manager._get_conn_stats(target_ns2)
    assert additional_synapses == 2
    assert stats.metype_cell_syn_average["metype-x"] == 1
    assert stats.metype_cell_syn_average["metype-y"] == 2

    # If we reinitialize the conn_manager and stats object then we should get the sum
    stats = DryRunStats()
    stats.pop_metype_gids = {"pop-A": {"metype-x": [0, 1], "metype-y": [2]}}
    conn_manager = ConnectionManagerBase(None, None, cell_manager, None, dry_run_stats=stats)
    conn_manager._synapse_reader = SonataReader(sonata_file, "NodeA__NodeA__chemical")

    total_synapses = conn_manager._get_conn_stats(target_ns2)
    assert total_synapses == total_synapses_metype_x + additional_synapses
    assert stats.metype_cell_syn_average["metype-x"] == 1
    assert stats.metype_cell_syn_average["metype-y"] == 2
