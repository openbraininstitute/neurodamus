import numpy as np
import numpy.testing as npt
import pytest

from tests.utils import check_signal_peaks

from .conftest import RINGTEST_DIR
from neurodamus import Neurodamus
from neurodamus.gap_junction import GapJunctionManager


def test_gapjunction_sonata_reader():
    from neurodamus.gap_junction import GapJunctionSynapseReader
    sonata_file = str(RINGTEST_DIR / "local_edges_C_electrical.h5")
    sonata_reader = GapJunctionSynapseReader.create(sonata_file)
    syn_params_sonata = sonata_reader._load_synapse_parameters(1)
    ref_junction_id_pre = np.array([2])
    ref_junction_id_post = np.array([0])
    ref_weight = np.array([100])
    npt.assert_allclose(syn_params_sonata.efferent_junction_id, ref_junction_id_pre)
    npt.assert_allclose(syn_params_sonata.afferent_junction_id, ref_junction_id_post)
    npt.assert_allclose(syn_params_sonata.weight, ref_weight)


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "network": str(RINGTEST_DIR / "circuit_config_gj.json"),
            "node_set": "ABC",
            "target_simulator": "NEURON",
            "inputs": {
                "Stimulus": {
                    "module": "pulse",
                    "input_type": "current_clamp",
                    "delay": 5,
                    "duration": 50,
                    "node_set": "RingC",
                    "amp_start": 10,
                    "width": 1,
                    "frequency": 50,
                }
            },
        }
    }
], indirect=True)
def test_gapjunctions(create_tmp_simulation_config_file):
    nd = Neurodamus(create_tmp_simulation_config_file)
    cell_manager = nd.circuits.get_node_manager("RingC")
    gids = cell_manager.get_final_gids()
    npt.assert_allclose(gids, np.array([2001, 2002, 2003]))

    gj_manager = nd.circuits.get_edge_manager("RingC", "RingC", GapJunctionManager)
    # Ensure we got our GJ instantiated and bi-directional
    gjs_1 = list(gj_manager.get_connections(2001))
    assert len(gjs_1) == 1
    assert gjs_1[0].sgid == 2003
    gjs_2 = list(gj_manager.get_connections(2002))
    assert len(gjs_2) == 1
    assert gjs_2[0].sgid == 2001
    gjs_3 = list(gj_manager.get_connections(2003))
    assert len(gjs_3) == 1
    assert gjs_3[0].sgid == 2002

    # Assert simulation went well
    # Check voltages
    from neuron import h
    tgt_cell = cell_manager.get_cell(2001)
    src_cell = cell_manager.get_cell(2003)
    tgtvar_vec = srcvar_vec = h.Vector()
    tgtvar_vec.record(tgt_cell._cellref.soma[0](0.5)._ref_v)
    srcvar_vec.record(src_cell._cellref.soma[0](0.5)._ref_v)

    h.finitialize()  # reinit for the recordings to be registered

    nd.run()

    npt.assert_allclose(tgtvar_vec.as_numpy(), srcvar_vec.as_numpy)
    check_signal_peaks(tgtvar_vec, [52, 57, 252, 257, 452, 457])
