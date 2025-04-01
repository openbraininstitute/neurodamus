from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from tests.utils import check_signal_peaks

from .conftest import RINGTEST_DIR
from neurodamus import Neurodamus
from neurodamus.connection_manager import SynapseRuleManager
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
def test_gapjunctions(create_tmp_simulation_config_file):
    nd = Neurodamus(create_tmp_simulation_config_file)
    cell_manager = nd.circuits.get_node_manager("RingC")
    gids = cell_manager.get_final_gids()
    npt.assert_allclose(gids, np.array([2001, 2002, 2003]))

    # chemical connections RingC -> RingC
    chemical_manager = nd.circuits.get_edge_manager("RingC", "RingC", SynapseRuleManager)
    assert len(list(chemical_manager.all_connections())) == 3

    # gap junction connections
    gj_manager = nd.circuits.get_edge_manager("RingC", "RingC", GapJunctionManager)
    assert len(list(gj_manager.all_connections())) == 2
    # Ensure we got our GJ instantiated and bi-directional
    gjs_1 = list(gj_manager.get_connections(2001))
    assert len(gjs_1) == 1
    assert gjs_1[0].sgid == 2003
    gjs_2 = list(gj_manager.get_connections(2003))
    assert len(gjs_2) == 1
    assert gjs_2[0].sgid == 2001

    # Assert simulation went well
    # Check voltages
    from neuron import h

    tgt_cell = cell_manager.get_cell(2001)
    tgtvar_vec = h.Vector()
    tgtvar_vec.record(tgt_cell._cellref.soma[0](0.5)._ref_v)

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
                    ),
                }
            }
        }
    ],
    indirect=True,
)
def test_gap_junction_corrections(capsys, create_tmp_simulation_config_file):
    """ test for gap junction calibration, the steps tested are similar to the thalamus publication
    """
    from neurodamus.core.configuration import SimConfig

    Neurodamus(create_tmp_simulation_config_file)

    assert SimConfig.beta_features == {
        "gapjunction_target_population": "RingC",
        "deterministic_stoch": True,
        "procedure_type": "validation_sim",
        "gjc": 0.2,
        "load_g_pas_file": str(
            RINGTEST_DIR / "gapjunctions" / "test_g_pas_passive.hdf5"
            ),
        "manual_MEComboInfo_file": str(
            RINGTEST_DIR / "gapjunctions" / "test_holding_per_gid.hdf5"
            ),
        }

    import re

    captured = capsys.readouterr()

    ref = re.compile(
        r"[\s\S]*Load user modification.*(CellDistributor: RingC).*\n"
        r".*Set deterministic = 1 for StochKv\n"
        r".*Set GJc = 0.2 for 2 gap synapses\n"
        r".*Update g_pas to fit 0.2 -.*for 1 cells\n"
        r".*Load holding_ic from manual_MEComboInfoFile.*for 1 cells\n[\s\S]*"
    )
    assert ref.match(captured.out)


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
                    "vc_amp": str(
                        RINGTEST_DIR / "gapjunctions" / "test_holding_voltage.hdf5"
                    )
                }
            }
        }
    ],
    indirect=True,
)
def test_gap_junction_corrections_otherfeatures(capsys, create_tmp_simulation_config_file):
    """ test the other features for gap junction calibration
    """
    from neurodamus.core.configuration import SimConfig

    Neurodamus(create_tmp_simulation_config_file)

    assert SimConfig.beta_features == {
        "gapjunction_target_population": "RingC",
        "remove_channels": "all",
        "procedure_type": "find_holding_current",
        "gjc": 0.2,
        "vc_amp": str(
            RINGTEST_DIR / "gapjunctions" / "test_holding_voltage.hdf5"
            )
        }

    import re

    captured = capsys.readouterr()

    ref = re.compile(
        r"[\s\S]*Load user modification.*(CellDistributor: RingC).*\n"
        r".*Set GJc = 0.2 for 2 gap synapses\n"
        r".*Remove channels type = all\n"
        r".*Inject V_Clamp without disabling holding current!\n"
        r".*Inject holding voltage from file.*for 1 cells\n"
        r".*Saving SEClamp Data\n[\s\S]*"
    )
    assert ref.match(captured.out)

    saved_seclamp = Path(SimConfig.output_root) / "data_for_host_0.p"
    assert saved_seclamp.exists()
