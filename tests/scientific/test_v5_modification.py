from pathlib import Path
import pytest
from neurodamus.utils.logging import log_verbose
# !! NOTE: Please dont import Neuron or Nd objects. pytest will trigger Neuron instantiation!

from tests.conftest import V5_SONATA

@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
        "src_dir": str(V5_SONATA),
        "simconfig_file": "simulation_config_mini.json",
        "extra_config": {
            "conditions": {
                "modifications": [
                    {
                        "name": "applyTTX",
                        "node_set": "Mini5",
                        "type": "ttx"
                    }
                ]
            },
        }
        }
    ],
    indirect=True,
)
def test_ttx_modification(create_tmp_simulation_config_file):
    """
    A test of enabling TTX with a short simulation.
    Expected outcome is non-zero spike count without TTX, zero with TTX.

    We require launching with mpiexec (numprocs=1).
    """
    from neurodamus.core import NeuronWrapper as Nd
    from neurodamus.core.configuration import GlobalConfig, LogLevel, SimConfig
    from neurodamus.node import Node

    GlobalConfig.verbosity = LogLevel.VERBOSE
    n = Node(create_tmp_simulation_config_file)

    # setup sim
    n.load_targets()
    n.create_cells()
    n.create_synapses()
    n.enable_stimulus()
    n.sim_init()
    n.solve()
    # _spike_vecs is a list of (spikes, ids)
    nspike_no_ttx = sum(len(spikes) for spikes, _ in n._spike_vecs)

    # setup sim again
    Nd.t = 0.0
    n._sim_ready = False
    n.enable_modifications()
    n.sim_init()
    n.solve()
    nspike_ttx = sum(len(spikes) for spikes, _ in n._spike_vecs)

    log_verbose("spikes without TTX = %s, with TTX = %s", nspike_no_ttx, nspike_ttx)
    assert nspike_no_ttx > 0
    assert nspike_ttx == 0

@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
        "src_dir": str(V5_SONATA),
        "simconfig_file": "simulation_config_mini.json",
        "extra_config": {
            "conditions": {
                "modifications": [
                    {
                        "name": "no_SK_E2",
                        "node_set": "Mini5",
                        "type": "configure_all_sections",
                        "section_configure": "%s.gSK_E2bar_SK_E2 = 0"
                    }
                ]
            },
        }
        }
    ],
    indirect=True,
)
def test_configure_all_sections_modification(create_tmp_simulation_config_file):
    """
    A test of performing configure_all_sections with a short simulation.
    Expected outcome is higher spike count when enabled.

    We require launching with mpiexec (numprocs=1).
    """
    from neurodamus.core import NeuronWrapper as Nd
    from neurodamus.core.configuration import GlobalConfig, LogLevel, SimConfig
    from neurodamus.node import Node

    GlobalConfig.verbosity = LogLevel.VERBOSE
    n = Node(create_tmp_simulation_config_file)

    # setup sim
    n.load_targets()
    n.create_cells()
    n.create_synapses()
    n.enable_stimulus()
    n.sim_init()
    n.solve(tstop=150)  # longer duration to see influence on spikes
    nspike_no_configure_all_sections = sum(len(spikes) for spikes, _ in n._spike_vecs)

    # setup sim again
    Nd.t = 0.0
    n._sim_ready = False
    n.enable_modifications()
    n.sim_init()
    n.solve(tstop=150)
    nspike_configure_all_sections = sum(len(spikes) for spikes, _ in n._spike_vecs)

    log_verbose("spikes without configure_all_sections = %s, with configure_all_sections = %s",
                nspike_configure_all_sections, nspike_no_configure_all_sections)
    assert (nspike_configure_all_sections > nspike_no_configure_all_sections)
