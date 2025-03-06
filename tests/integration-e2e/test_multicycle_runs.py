import numpy as np
import pytest
from pathlib import Path

SIM_DIR = Path(__file__).parent.parent.absolute() / "simulations"


def test_nodeset_target_generate_subtargets():
    from neurodamus.core.nodeset import NodeSet
    from neurodamus.target_manager import NodesetTarget

    N_PARTS = 3
    raw_gids_a = list(range(10))
    raw_gids_b = list(range(5))
    nodes_popA = NodeSet(raw_gids_a).register_global("pop_A")
    nodes_popB = NodeSet(raw_gids_b).register_global("pop_B")
    target = NodesetTarget("Column", [nodes_popA, nodes_popB])
    assert np.array_equal(target.get_gids(),
                          np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1000, 1001, 1002, 1003, 1004]))

    subtargets = target.generate_subtargets(N_PARTS)
    assert len(subtargets) == N_PARTS
    assert len(subtargets[0]) == 2
    subtarget_popA = subtargets[0][0]
    subtarget_popB = subtargets[0][1]
    assert subtarget_popA.name == "pop_A__Column_0"
    assert subtarget_popA.population_names == {"pop_A"}
    assert np.array_equal(subtarget_popA.get_gids(), np.array([0, 3, 6, 9]))
    assert subtarget_popB.name == "pop_B__Column_0"
    assert subtarget_popB.population_names == {"pop_B"}
    assert np.array_equal(subtarget_popB.get_gids(), np.array([1000, 1003]))
    assert np.array_equal(subtargets[1][0].get_gids(), np.array([1, 4, 7]))
    assert np.array_equal(subtargets[2][0].get_gids(), np.array([2, 5, 8]))
    assert np.array_equal(subtargets[1][1].get_gids(), np.array([1001, 1004]))
    assert np.array_equal(subtargets[2][1].get_gids(), np.array([1002]))


def _read_sonata_spike_file(spike_file):
    import libsonata
    spikes = libsonata.SpikeReader(spike_file)
    pop_name = spikes.get_population_names()[0]
    data = spikes[pop_name].get()
    timestamps = np.array([x[1] for x in data])
    spike_gids = np.array([x[0] for x in data])
    return timestamps, spike_gids


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "src_dir": str(SIM_DIR / "v5_sonata"),
        "simconfig_file": "simulation_config_mini.json",
        "extra_config": {
            "target_simulator": "CORENEURON",
        }
    }
], indirect=True)
def test_v5_sonata_multisteps(create_tmp_simulation_config_file):
    import numpy.testing as npt
    from neurodamus import Neurodamus

    print("asdf")
    config_file = create_tmp_simulation_config_file

    nd = Neurodamus(config_file, modelbuilding_steps=3)
    nd.run()

    # compare spikes with refs
    spike_gids = np.array([
        4, 2, 0
    ])  # 0-based
    # timestamps = np.array([
    #     33.425, 37.35, 39.725
    # ])
    spike_file = Path(nd._run_conf.get("OutputRoot"))/nd._run_conf.get("SpikesFile")
    obtained_timestamps, obtained_spike_gids = _read_sonata_spike_file(spike_file)
    npt.assert_allclose(spike_gids, obtained_spike_gids)
    # coreneuron and neuron have a discrepancy now:
    # https://github.com/openbraininstitute/neurodamus/issues/44?issue=openbraininstitute%7Cneurodamus%7C3
    # TODO: re-enable once it is fixed
    # npt.assert_allclose(timestamps, obtained_timestamps)



@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "src_dir": str(SIM_DIR / "usecase3"),
        "simconfig_file": "simulation_sonata_coreneuron.json"
    }
], indirect=True)
def test_usecase3_sonata_multisteps(create_tmp_simulation_config_file):
    import numpy.testing as npt
    from neurodamus import Neurodamus

    config_file = create_tmp_simulation_config_file
    nd = Neurodamus(config_file, modelbuilding_steps=2)
    nd.run()

    # compare spikes with refs
    spike_gids = np.array([
        0, 1, 2, 0, 1, 2, 0, 0, 1, 2, 2, 2, 2, 2, 0, 2, 1, 0, 2, 0, 1
    ])  # 0-based
    timestamps = np.array([
        0.2, 0.3, 0.3, 2.5, 3.4, 4.2, 5.5, 7., 7.4, 8.6, 13.8, 19.6, 25.7, 32., 36.4, 38.5,
        40.8, 42.6, 45.2, 48.3, 49.9
    ])
    spike_file = Path(nd._run_conf.get("OutputRoot"))/nd._run_conf.get("SpikesFile")
    obtained_timestamps, obtained_spike_gids = _read_sonata_spike_file(spike_file)
    npt.assert_allclose(spike_gids, obtained_spike_gids)
    npt.assert_allclose(timestamps, obtained_timestamps)
