import pytest
import numpy.testing as npt
from tests import utils

from .conftest import RINGTEST_DIR

INPUT_SPIKES_FILE = str(RINGTEST_DIR / "input_spikes.h5")


@pytest.fixture
def ringtest_virtual_pop_config():
    from tempfile import NamedTemporaryFile
    import os
    import json

    circuit_config_data = {
        "version": 2,
        "networks": {
            "nodes": [
                {
                    "nodes_file": str(RINGTEST_DIR / "nodes_A.h5"),
                    "populations": {
                        "RingA": {
                            "type": "virtual",
                        }
                    }
                },
                {
                    "nodes_file": str(RINGTEST_DIR / "nodes_B.h5"),
                    "populations": {
                        "RingB": {
                            "type": "biophysical",
                            "morphologies_dir": str(RINGTEST_DIR / "morphologies/swc"),
                            "biophysical_neuron_models_dir": str(RINGTEST_DIR / "hoc"),
                            "alternate_morphologies": {
                                "neurolucida-asc": str(RINGTEST_DIR / "morphologies/asc")
                            }
                        }
                    }
                }
            ],
            "edges": [
                {
                    "edges_file": str(RINGTEST_DIR / "local_edges_B.h5"),
                    "populations": {
                        "RingB__RingB__chemical": {
                            "type": "chemical"
                        }
                    }
                },
                {
                    "edges_file": str(RINGTEST_DIR / "edges_AB.h5"),
                    "populations": {
                        "RingA__RingB__chemical": {
                            "type": "chemical"
                        }
                    }
                }
            ]
        }
    }

    with NamedTemporaryFile("w", suffix='.json', delete=False) as config_file:
        json.dump(circuit_config_data, config_file)

    yield dict(
        network=config_file.name,
        node_sets_file=str(RINGTEST_DIR / "nodesets.json"),
        target_simulator="NEURON",
        run={
            "random_seed": 1122,
            "dt": 0.1,
            "tstop": 50,
        },
        node_set="Mosaic",
        conditions={
            "celsius": 35,
            "v_init": -65
        }
    )

    os.unlink(config_file.name)


@pytest.mark.forked
def test_sonata_spikes_reader():
    from neurodamus.replay import SpikeManager, MissingSpikesPopulationError

    timestamps, spike_gids = SpikeManager._read_spikes_sonata(INPUT_SPIKES_FILE, "RingA")
    npt.assert_allclose(timestamps, [0.1, 0.15, 0.175, 2.275, 3.025, 3.45, 4.35, 5.7, 6.975, 7.725])
    npt.assert_equal(spike_gids, [1, 3, 2, 1, 2, 3, 1, 2, 3, 1])

    # We do an internal assertion when the population doesnt exist. Verify it works as expected
    with pytest.raises(MissingSpikesPopulationError, match="Spikes population not found"):
        SpikeManager._read_spikes_sonata(INPUT_SPIKES_FILE, "wont-exist")


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "inputs": {
                "spikeReplay" : {
                    "module": "synapse_replay",
                    "input_type": "spikes",
                    "spike_file": INPUT_SPIKES_FILE,
                    "delay": 0,
                    "duration": 50,
                    "node_set": "RingB",
                }
            }
        }
    },
], indirect=True)
@pytest.mark.forked
def test_sonata_parse_synapse_replay_input(create_tmp_simulation_config_file):
    from neurodamus.core.configuration import SimConfig

    SimConfig.init(create_tmp_simulation_config_file, {})

    spikes_replay = SimConfig.stimuli["spikeReplay"]
    assert spikes_replay['Target'] == 'RingB'
    assert spikes_replay['Mode'] == 'Current'
    assert spikes_replay['Pattern'] == 'SynapseReplay'
    assert spikes_replay['Delay'] == 0.0
    assert spikes_replay['Duration'] == 50.0
    assert spikes_replay['SpikeFile'] == INPUT_SPIKES_FILE


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "inputs": {
                "spikeReplay" : {
                    "module": "synapse_replay",
                    "input_type": "spikes",
                    "spike_file": INPUT_SPIKES_FILE,
                    "delay": 0,
                    "duration": 50,
                    "node_set": "RingB",
                }
            }
        }
    },
], indirect=True)
@pytest.mark.forked
def test_replay_stim_generated_run(create_tmp_simulation_config_file):
    from neurodamus import Neurodamus
    from neurodamus.core import NeurodamusCore as Nd
    from neurodamus.core.configuration import Feature
    from neurodamus.connection import NetConType

    nd = Neurodamus(
        create_tmp_simulation_config_file,
        restrict_features=[Feature.Replay],
        disable_reports=True,
        cleanup_atexit=False,
    )

    edges_a_b = nd.circuits.get_edge_manager("RingA", "RingB")
    conn_1_1001 = next(edges_a_b.get_connections(1001, 1))
    time_vec = conn_1_1001._replay.time_vec.as_numpy()
    npt.assert_allclose(time_vec, [0.1, 2.275, 4.35, 7.725])

    # get the netcon for connection 1->1001
    cell = nd._pc.gid2cell(1001)
    pre_cell = nd._pc.gid2cell(1)
    nclist = Nd.cvode.netconlist(pre_cell, cell, "")
    assert nclist.count() == 1
    netcon1 = nclist.o(0)
    assert netcon1.weight[4] == int(NetConType.NC_PRESYN)

    # get the netcon generated by ReplayStim for connection 1->1001
    nclist = Nd.cvode.netconlist("", cell, "")
    replay_netcons = []
    for netcon in nclist:
        if netcon.srcgid() < 0 and netcon.weight[4] == int(NetConType.NC_REPLAY):
            replay_netcons += [netcon]
            assert netcon.weight[0] == netcon1.weight[0]
            assert netcon.delay == netcon1.delay
            assert netcon.threshold == 10
    assert len(replay_netcons) > 0

    # get voltage variations in cell 1001
    cell = nd.circuits.get_node_manager("RingB").get_cell(1001)
    voltage_vec = Nd.Vector()
    voltage_vec.record(cell._cellref.soma[0](0.5)._ref_v)

    Nd.finitialize()
    nd.run()

    utils.check_signal_peaks(voltage_vec, [42, 84], 0.5)


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_virtual_pop_config",
        "extra_config": {
            "inputs": {
                "spikeReplay" : {
                    "module": "synapse_replay",
                    "input_type": "spikes",
                    "spike_file": INPUT_SPIKES_FILE,
                    "delay": 0,
                    "duration": 50,
                    "node_set": "RingB",
                }
            }
        }
    },
], indirect=True)
@pytest.mark.forked
def test_replay_virtual_population(create_tmp_simulation_config_file):
    from neurodamus import Neurodamus
    from neurodamus.core import NeurodamusCore as Nd
    from neurodamus.core.configuration import Feature
    from neurodamus.connection import NetConType

    nd = Neurodamus(
        create_tmp_simulation_config_file,
        restrict_features=[Feature.Replay],
        disable_reports=True,
        cleanup_atexit=False,
    )

    edges_a_b = nd.circuits.get_edge_manager("RingA", "RingB")
    conn_1_1001 = next(edges_a_b.get_connections(1001, 1))
    time_vec = conn_1_1001._replay.time_vec.as_numpy()
    npt.assert_allclose(time_vec, [0.1, 2.275, 4.35, 7.725])
    # # get the netcon generated by ReplayStim for connection 1->1001
    cell = nd._pc.gid2cell(1001)
    nclist = Nd.cvode.netconlist("", cell, "")

    replay_netcons = []
    for netcon in nclist:
        if netcon.srcgid() < 0 and netcon.weight[4] == int(NetConType.NC_REPLAY):
            replay_netcons += [netcon]
    assert len(replay_netcons) > 0

    # get voltage variations in cell 1001
    cell = nd.circuits.get_node_manager("RingB").get_cell(1001)
    voltage_vec = Nd.Vector()
    voltage_vec.record(cell._cellref.soma[0](0.5)._ref_v)

    Nd.finitialize()
    nd.run()

    utils.check_signal_peaks(voltage_vec, [42, 84], 0.5)
