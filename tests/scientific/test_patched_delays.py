from pathlib import Path

import pytest

from neurodamus.core.configuration import GlobalConfig, LogLevel, SimConfig
from neurodamus.io.sonata_config import ConnectionOverride
from neurodamus.node import Node

SIM_DIR = Path(__file__).parent.parent.absolute() / "simulations" / "sscx-v7-plasticity"
CONFIG_FILE = SIM_DIR / "simulation_config_base.json"


@pytest.mark.slow
def test_eager_caching():
    """
    A test of the impact of eager caching of synaptic parameters. BBPBGLIB-813
    """
    from neurodamus.core import NeuronWrapper as Nd

    # create Node from config
    GlobalConfig.verbosity = LogLevel.VERBOSE
    n = Node(str(CONFIG_FILE))

    # append Connection blocks programmatically
    # plasticity
    CONN_plast = ConnectionOverride(
        name="plasticity",
        source="pre_L5_PCs",
        destination="post_L5_PC",
        weight=1.0,
        modoverride="GluSynapse",
    )
    SimConfig.connections[CONN_plast.name] = CONN_plast
    # init_I_E
    CONN_i2e = ConnectionOverride(
        name="init_I_E", source="pre_L5_BC", destination="post_L5_PC", weight=1.0
    )
    SimConfig.connections[CONN_i2e.name] = CONN_i2e
    assert len(SimConfig.connections) == 2

    # setup sim
    n.load_targets()
    n.create_cells()
    n.create_synapses()
    # manually finalize synapse managers (otherwise netcons are not created)
    for syn_manager in n._circuits.all_synapse_managers():
        syn_manager.finalize(n._run_conf.get("BaseSeed", 0))
    n.sim_init()  # not really necessary

    # here we get the HOC object for the post cell
    tgt = n._target_manager.get_target("post_L5_PC")
    post_gid = tgt.gids(raw_gids=True)[0]
    post_cell = n.circuits.global_manager.get_cellref(post_gid)
    # here we check that all synaptic delays are rounded to timestep
    # we skip minis netcons (having precell == None)
    delays = [
        nc.delay
        for nc in Nd.h.cvode.netconlist("", post_cell, "")
        if nc.precell() is not None
    ]
    patch_delays = [int(x / Nd.dt + 1e-5) * Nd.dt for x in delays]
    assert delays == patch_delays
