import re

import numpy as np
import numpy.testing as npt
import pytest

from tests.utils import check_signal_peaks

from .conftest import RINGTEST_DIR
from neurodamus import Neurodamus
from neurodamus.connection_manager import SynapseRuleManager
from neurodamus.core.configuration import SimConfig
from neurodamus.gap_junction import GapJunctionManager


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
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
                }
            }
        }
    ],
    indirect=True,
)
def test_gapjunctions_default(create_tmp_simulation_config_file):
    """Test general simulation flow including gap junctions, without any user correction"""

    nd = Neurodamus(create_tmp_simulation_config_file)

    cell_manager = nd.circuits.get_node_manager("RingC")
    offset = cell_manager.local_nodes.offset
    assert offset == 2000
    gids = cell_manager.get_final_gids()
    npt.assert_allclose(gids, np.array([2001, 2002, 2003]))

    # chemical connections RingC -> RingC
    chemical_manager = nd.circuits.get_edge_manager("RingC", "RingC", SynapseRuleManager)
    assert len(list(chemical_manager.all_connections())) == 3

    # gap junction connections
    gj_manager = nd.circuits.get_edge_manager("RingC", "RingC", GapJunctionManager)
    assert len(list(gj_manager.all_connections())) == 2
    # Ensure we got our GJ instantiated and bi-directional
    gjs_1 = list(gj_manager.get_connections(post_gids=2001, pre_gids=2003))
    assert len(gjs_1) == 1
    gjs_2 = list(gj_manager.get_connections(post_gids=2003, pre_gids=2001))
    assert len(gjs_2) == 1

    # check gap junction parameters values without user correction
    tgt_cellref = cell_manager.getCell(2001)
    connection = next(gj_manager.get_connections(post_gids=[2003], pre_gids=[2001]))
    assert connection.synapses[0].g == 100
    assert tgt_cellref.soma[0](0.5).pas.g == 0
    assert gj_manager.holding_ic_per_gid is None
    assert gj_manager.seclamp_per_gid is None

    # Assert simulation went well
    # Check voltages
    from neuron import h

    tgtvar_vec = h.Vector()
    tgtvar_vec.record(tgt_cellref.soma[0](0.5)._ref_v)
    h.finitialize()  # reinit for the recordings to be registered
    nd.run()
    check_signal_peaks(tgtvar_vec, [52, 57, 252, 257, 452, 457])


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {
                "network": str(RINGTEST_DIR / "circuit_config_gj.json"),
                "node_set": "ABC",
                "beta_features": {
                    "gapjunction_target_population": "RingC",
                    "deterministic_stoch": True,
                    "procedure_type": "validation_sim",
                    "gjc": 0.2,
                    "load_g_pas_file": str(
                        RINGTEST_DIR / "gapjunctions" / "test_g_pas_passive.hdf5"
                    ),
                    "manual_MEComboInfo_file": str(
                        RINGTEST_DIR / "gapjunctions" / "test_holding_per_gid.hdf5"
                    )
                }
            }
        }
    ],
    indirect=True,
)
def test_gap_junction_corrections(capsys, create_tmp_simulation_config_file):
    """Test gap junction calibration designed by Oren,
    the steps are similar to the thalamus publication"""

    nd = Neurodamus(create_tmp_simulation_config_file)

    assert SimConfig.beta_features == {
        "gapjunction_target_population": "RingC",
        "deterministic_stoch": True,
        "procedure_type": "validation_sim",
        "gjc": 0.2,
        "load_g_pas_file": str(RINGTEST_DIR / "gapjunctions" / "test_g_pas_passive.hdf5"),
        "manual_MEComboInfo_file": str(RINGTEST_DIR / "gapjunctions" / "test_holding_per_gid.hdf5"),
    }

    # check log
    captured = capsys.readouterr()
    ref = re.compile(
        r"[\s\S]*Load user modification.*(CellDistributor: RingC).*\n"
        r".*Set deterministic = 1 for StochKv\n"
        r".*Set GJc = 0.2 for 2 gap synapses\n"
        r".*Update g_pas to fit 0.2 -.*for 1 cells\n"
        r".*Load holding_ic from manual_MEComboInfoFile.*for 1 cells\n[\s\S]*"
    )
    assert ref.match(captured.out)

    # check gap junction parameters after user correction
    cell_manager = nd.circuits.get_node_manager("RingC")
    tgt_cellref = cell_manager.get_cellref(2001)
    gj_manager = nd.circuits.get_edge_manager("RingC", "RingC", GapJunctionManager)
    connection = next(gj_manager.get_connections(post_gids=[2003], pre_gids=[2001]))
    assert connection.synapses[0].g == 0.2
    assert tgt_cellref.soma[0](0.5).pas.g == 0.033
    assert len(gj_manager.holding_ic_per_gid) == 1
    iclamp = gj_manager.holding_ic_per_gid[2001]
    assert "IClamp" in iclamp.hname()
    npt.assert_allclose(iclamp.dur, 9e9)
    npt.assert_allclose(iclamp.amp, -0.00108486, rtol=1e-5)
    assert gj_manager.seclamp_per_gid == {}

    # Assert simulation went well
    nd.run()


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {
                "network": str(RINGTEST_DIR / "circuit_config_gj.json"),
                "node_set": "ABC",
                "beta_features": {
                    "gapjunction_target_population": "RingC",
                    "remove_channels": "all",
                    "procedure_type": "find_holding_current",
                    "gjc": 0.2,
                    "vc_amp": str(RINGTEST_DIR / "gapjunctions" / "test_holding_voltage.hdf5"),
                }
            }
        }
    ],
    indirect=True,
)
def test_gap_junction_corrections_otherfeatures(capsys, create_tmp_simulation_config_file):
    """Test the other steps for the gap junction calibration designed by Oren,
    but not used in the publication"""

    nd = Neurodamus(create_tmp_simulation_config_file)

    assert SimConfig.beta_features == {
        "gapjunction_target_population": "RingC",
        "remove_channels": "all",
        "procedure_type": "find_holding_current",
        "gjc": 0.2,
        "vc_amp": str(RINGTEST_DIR / "gapjunctions" / "test_holding_voltage.hdf5"),
    }

    # check log
    captured = capsys.readouterr()
    ref = re.compile(
        r"[\s\S]*Load user modification.*(CellDistributor: RingC).*\n"
        r".*Set GJc = 0.2 for 2 gap synapses\n"
        r".*Remove channels type = all\n"
        r".*Inject V_Clamp without disabling holding current!\n"
        r".*Inject holding voltage from file.*for 1 cells\n[\s\S]*"
    )
    assert ref.match(captured.out)

    # check holding current
    gj_manager = nd.circuits.get_edge_manager("RingC", "RingC", GapJunctionManager)
    assert gj_manager.holding_ic_per_gid == {}
    assert len(gj_manager.seclamp_per_gid) == 1
    assert "SEClamp" in gj_manager.seclamp_per_gid[2001].hname()
    seclamp = gj_manager.seclamp_per_gid[2001]
    npt.assert_allclose(seclamp.dur1, 9e9)
    npt.assert_allclose(seclamp.amp1, 0.1)
    npt.assert_allclose(seclamp.rs, 0.0000001)

    # Assert simulation went well
    nd.run()
