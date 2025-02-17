import pytest
import numpy.testing as npt


@pytest.fixture
def input_spikes_file(ringtest_dir):
    return str(ringtest_dir / "input_spikes.h5")


@pytest.fixture
def replay_sim_config_file(ringtest_baseconfig, input_spikes_file):
    from tempfile import NamedTemporaryFile
    import json

    ringtest_baseconfig["inputs"] = {}
    ringtest_baseconfig["inputs"]["spikeReplay"] = {
        "module": "synapse_replay",
        "input_type": "spikes",
        "spike_file": input_spikes_file,
        "delay": 0,
        "duration": 100,
        "node_set": "Mosaic",  # no limits!
    }

    with NamedTemporaryFile("w", suffix='.json', delete=False) as config_file:
        json.dump(ringtest_baseconfig, config_file)
    return config_file


@pytest.mark.forked
def test_sonata_spikes_reader(input_spikes_file):
    from neurodamus.replay import SpikeManager, MissingSpikesPopulationError

    timestamps, spike_gids = SpikeManager._read_spikes_sonata(input_spikes_file, "RingA")
    npt.assert_allclose(timestamps, [0.1, 0.15, 0.175, 2.275, 3.025, 3.45, 4.35, 5.7, 6.975, 7.725])
    npt.assert_equal(spike_gids, [1, 3, 2, 1, 2, 3, 1, 2, 3, 1])

    # We do an internal assertion when the population doesnt exist. Verify it works as expected
    with pytest.raises(MissingSpikesPopulationError, match="Spikes population not found"):
        SpikeManager._read_spikes_sonata(input_spikes_file, "wont-exist")


@pytest.mark.forked
def test_synapse_replay_load(replay_sim_config_file, input_spikes_file):
    from neurodamus import Neurodamus
    from neurodamus.core.configuration import Feature
    from neurodamus.core.configuration import SimConfig

    Neurodamus(
        replay_sim_config_file.name,
        restrict_node_populations=["RingA"],
        restrict_features=[Feature.Replay],
        restrict_connectivity=1,  # base restriction, no projections
        disable_reports=True,
        cleanup_atexit=False,
    )

    spikes_replay = SimConfig.stimuli["spikeReplay"]
    assert spikes_replay['Target'] == 'Mosaic'
    assert spikes_replay['Mode'] == 'Current'
    assert spikes_replay['Pattern'] == 'SynapseReplay'
    assert spikes_replay['Delay'] == 0.0
    assert spikes_replay['Duration'] == 100.0
    assert spikes_replay['SpikeFile'] == input_spikes_file
