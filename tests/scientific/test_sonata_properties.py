from neurodamus.node import Node
from neurodamus.core.configuration import GlobalConfig, SimConfig, LogLevel
from pathlib import Path

SIM_DIR = Path(__file__).parent.parent.absolute() / "simulations" / "sscx-v7-plasticity"
CONFIG_FILE = SIM_DIR / "simulation_config_base.json"


def test_input_resistance():
    """
    A test of getting input resistance values from SONATA nodes file. BBPBGLIB-806
    """
    # create Node from config
    GlobalConfig.verbosity = LogLevel.VERBOSE
    n = Node(str(CONFIG_FILE))

    # append Stimulus and StimulusInject blocks programmatically
    # relativeOU
    STIM_relativeOU = {
        "Mode": "Conductance",
        "Pattern": "RelativeOrnsteinUhlenbeck",
        "Delay": 50,
        "Duration": 200,
        "Reversal": 0,
        "Tau": 2.8,
        "MeanPercent": 20,
        "SDPercent": 20,
        "Name": "relativeOU",
        "Target": "L5_5cells"
    }
    SimConfig.stimuli.append(STIM_relativeOU)

    # setup sim
    n.load_targets()
    n.create_cells()
    n.create_synapses()
    n.enable_stimulus()

    # Ensure we have our targets and loaded cells ok
    target_small = n.target_manager.get_target("L5_5cells")
    cell_manager = n.circuits.get_node_manager("All")
    gids = cell_manager.local_nodes.final_gids()
    assert target_small.nodesets[0].offset == 0
    assert target_small.gid_count() == 5
    assert cell_manager.total_cells == 5
    assert len(cell_manager.local_nodes) == 5
    for gid in (1, 2, 3, 4, 5):
        assert gid in target_small
        assert gid in gids

    # manually finalize synapse managers (otherwise netcons are not created)
    for syn_manager in n._circuits.all_synapse_managers():
        syn_manager.finalize(n._run_conf.get("BaseSeed", 0))
    n.sim_init()
    n.solve()

    # check spikes
    nspike = sum(len(spikes) for spikes, _ in n._spike_vecs)
    assert nspike == 10


def test_input_resistance_2():
    """
    A test of getting input resistance values from SONATA nodes file. BBPBGLIB-806
    """
    # create Node from config
    GlobalConfig.verbosity = LogLevel.VERBOSE
    n = Node(str(CONFIG_FILE))

    # append Stimulus and StimulusInject blocks programmatically
    # relativeSN
    STIM_relativeSN = {
        "Mode": "Conductance",
        "Pattern": "RelativeShotNoise",
        "Delay": 50,
        "Duration": 200,
        "Reversal": 0,
        "RiseTime": 2.8,
        "DecayTime": 28,
        "RelativeSkew": 0.5,
        "MeanPercent": 20,
        "SDPercent": 20,
        "Name": "relativeSN",
        "Target": "L5_5cells"
    }
    SimConfig.stimuli.append(STIM_relativeSN)

    # setup sim
    n.load_targets()
    n.create_cells()
    n.create_synapses()
    n.enable_stimulus()
    # manually finalize synapse managers (otherwise netcons are not created)
    for syn_manager in n._circuits.all_synapse_managers():
        syn_manager.finalize(n._run_conf.get("BaseSeed", 0))
    n.sim_init()
    n.solve()

    # check spikes
    nspike = sum(len(spikes) for spikes, _ in n._spike_vecs)
    assert nspike == 6
