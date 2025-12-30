import json
import os
from tempfile import NamedTemporaryFile

import numpy as np
import numpy.testing as npt
import pytest
from scipy.signal import find_peaks

from tests.conftest import RINGTEST_DIR

from neurodamus import Neurodamus
from neurodamus.connection import NetConType
from neurodamus.core.configuration import ConfigurationError, Feature, SimConfig
from neurodamus.replay import SpikeManager, read_sonata_spikes

INPUT_SPIKES_FILE = str(RINGTEST_DIR / "input_spikes.h5")


@pytest.fixture
def ringtest_virtual_pop_config():
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

    with NamedTemporaryFile("w", suffix=".json", delete=False) as config_file:
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


def test_sonata_spikes_reader():
    timestamps, spike_gids = read_sonata_spikes(INPUT_SPIKES_FILE, "RingA")
    npt.assert_allclose(timestamps, [0.1, 0.15, 0.175, 2.275, 3.025, 3.45, 4.35, 5.7, 6.975, 7.725])
    npt.assert_equal(spike_gids, [0, 2, 1, 0, 1, 2, 0, 1, 2, 0])


def test_sonata_spike_manager_with_delay():
    spike_manager = SpikeManager(INPUT_SPIKES_FILE, delay=10, population="RingA")
    spike_events = spike_manager.get_map()
    spike_gids = spike_events.keys()
    npt.assert_equal(spike_gids, [0, 1, 2])
    npt.assert_allclose(spike_events.get(0), [10.1, 12.275, 14.35, 17.725])
    npt.assert_allclose(spike_manager.filter_map(pre_gids=[1, 2]).get(1), [10.175, 13.025, 15.7])


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "inputs": {
                "spikeReplay": {
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
    SimConfig.init(create_tmp_simulation_config_file, {})

    assert len(SimConfig.stimuli) == 1
    spikes_replay = SimConfig.stimuli[0]
    assert spikes_replay["Target"] == "RingB"
    assert spikes_replay["Mode"] == "Current"
    assert spikes_replay["Pattern"] == "SynapseReplay"
    assert spikes_replay["Delay"] == 0.0
    assert spikes_replay["Duration"] == 50.0
    assert spikes_replay["SpikeFile"] == INPUT_SPIKES_FILE


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "inputs": {
                "spikeReplay": {
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
    # the import of neuron must be at function level,
    # otherwise will impact other tests even done in forked processes
    from neurodamus.core import NeuronWrapper as Nd
    nd = Neurodamus(
        create_tmp_simulation_config_file,
        restrict_features=[Feature.Replay],
        disable_reports=True,
        cleanup_atexit=False,
    )

    edges_a_b = nd.circuits.get_edge_manager("RingA", "RingB")
    conn_0_1000 = next(edges_a_b.get_connections(1000, 0))
    time_vec = conn_0_1000._replay.time_vec.as_numpy()
    npt.assert_allclose(time_vec, [0.1, 2.275, 4.35, 7.725])

    # get the netcon for connection 0->1000
    cell = nd._pc.gid2cell(1000)
    pre_cell = nd._pc.gid2cell(0)
    nclist = Nd.cvode.netconlist(pre_cell, cell, "")
    assert nclist.count() == 1
    netcon1 = nclist.o(0)
    assert netcon1.weight[4] == int(NetConType.NC_PRESYN)

    # get the netcon generated by ReplayStim for connection 0->1000
    nclist = Nd.cvode.netconlist("", cell, "")
    replay_netcons = []
    for netcon in nclist:
        if netcon.srcgid() < 0 and netcon.weight[4] == int(NetConType.NC_REPLAY):
            replay_netcons += [netcon]
            assert netcon.weight[0] == netcon1.weight[0]
            assert netcon.delay == netcon1.delay
            assert netcon.threshold == 10
    assert len(replay_netcons) > 0

    # get voltage variations in cell 1000
    cell = nd.circuits.get_node_manager("RingB").get_cell(1000)
    voltage_vec = Nd.Vector()
    voltage_vec.record(cell._cellref.soma[0](0.5)._ref_v)

    Nd.finitialize()
    nd.run()

    peaks_pos = find_peaks(voltage_vec, prominence=0.5)[0]
    np.testing.assert_allclose(peaks_pos, [42, 84])


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_virtual_pop_config",
        "extra_config": {
            "inputs": {
                "spikeReplay": {
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
    from neurodamus.core import NeuronWrapper as Nd

    nd = Neurodamus(
        create_tmp_simulation_config_file,
        restrict_features=[Feature.Replay],
        disable_reports=True,
        cleanup_atexit=False,
    )

    edges_a_b = nd.circuits.get_edge_manager("RingA", "RingB")
    conn_0_1000 = next(edges_a_b.get_connections(1000, 0))
    time_vec = conn_0_1000._replay.time_vec.as_numpy()
    npt.assert_allclose(time_vec, [0.1, 2.275, 4.35, 7.725])
    # # get the netcon generated by ReplayStim for connection 1->1000
    cell = nd._pc.gid2cell(1000)
    nclist = Nd.cvode.netconlist("", cell, "")

    replay_netcons = []
    for netcon in nclist:
        if netcon.srcgid() < 0 and netcon.weight[4] == int(NetConType.NC_REPLAY):
            replay_netcons += [netcon]
    assert len(replay_netcons) > 0

    # get voltage variations in cell 1000
    cell = nd.circuits.get_node_manager("RingB").get_cell(1000)
    voltage_vec = Nd.Vector()
    voltage_vec.record(cell._cellref.soma[0](0.5)._ref_v)

    Nd.finitialize()
    nd.run()

    peaks_pos = find_peaks(voltage_vec, prominence=0.5)[0]
    np.testing.assert_allclose(peaks_pos, [42, 84])
