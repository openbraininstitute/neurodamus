import pytest

from neurodamus.core.configuration import SimConfig
from neurodamus.core.configuration import CellPermute

from neurodamus.core.coreneuron_simulation_config import CoreSimulationConfig


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [{"simconfig_fixture": "ringtest_baseconfig",
              "extra_config": {
            "target_simulator": "CORENEURON",
        }
      }],
    indirect=True,
)
def test_cli_cell_permute_simple_setting(create_tmp_simulation_config_file):
    from neurodamus import Neurodamus
    Neurodamus(create_tmp_simulation_config_file, cell_permute="node-adjacency", keep_build=True)
    assert SimConfig.cell_permute == CellPermute.NODE_ADJACENCY
    sim_conf = CoreSimulationConfig.load("build/sim.conf")
    assert sim_conf.cell_permute == 1

@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [{"simconfig_fixture": "ringtest_baseconfig",
              "extra_config": {
            "target_simulator": "CORENEURON",
        }
      }],
    indirect=True,
)
def test_cli_cell_permute_default(create_tmp_simulation_config_file):
    from neurodamus import Neurodamus
    Neurodamus(create_tmp_simulation_config_file, keep_build=True)
    assert SimConfig.cell_permute == CellPermute.UNPERMUTED
    sim_conf = CoreSimulationConfig.load("build/sim.conf")
    assert sim_conf.cell_permute == 0

