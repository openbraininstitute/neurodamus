import pytest
from pathlib import Path

from neurodamus.core.configuration import ConfigurationError
from neurodamus.io.synapse_reader import SynapseParameters
from neurodamus.node import Node

SIM_DIR = Path(__file__).parent.parent.absolute() / "simulations" / "v5_sonata"
CONFIG_FILE_MINI = SIM_DIR / "simulation_config_mini.json"


@pytest.mark.slow
# This test is to mimic the error reported in HPCTM-1687 during connection.add_syanpses()
# when detecting conn._synapse_params with more than one element is not None
def test_add_synapses():
    n = Node(str(CONFIG_FILE_MINI))
    n.load_targets()
    n.create_cells()
    n.create_synapses()
    syn_manager = n.circuits.get_edge_manager("external_default", "default")
    conn = list(syn_manager.get_connections(1))[0]
    new_params = SynapseParameters.create_array(1)
    n_syns = len(conn._synapse_params)
    assert n_syns > 1
    new_params[0].sgid = conn.sgid
    new_params[0].isec = 0
    conn.add_synapses(n._target_manager, new_params)
    assert len(conn._synapse_params) == n_syns + 1
    for syn_manager in n._circuits.all_synapse_managers():
        syn_manager.finalize(0, False)


@pytest.mark.slow
@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "sonata_config",
        "extra_config": {
            "connection_overrides": [
                {
                    "name": "config_ERR",
                    "source": "nodesPopB",
                    "target": "nodesPopB",
                    "synapse_configure": "%s.dummy=1"
                }
            ]
        }
    }
], indirect=True)
def test_config_error(create_tmp_simulation_config_file):
    with pytest.raises(ConfigurationError):
        n = Node(create_tmp_simulation_config_file)
        n.load_targets()
        n.create_cells()
        n.create_synapses()
        for syn_manager in n._circuits.all_synapse_managers():
            syn_manager.finalize(0, False)
