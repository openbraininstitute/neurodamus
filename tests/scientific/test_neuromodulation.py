import numpy as np
import numpy.testing as npt
import pytest
from pathlib import Path

SIM_DIR = Path(__file__).parent.parent.absolute() / "simulations" / "neuromodulation"


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "src_dir": str(SIM_DIR),
        "simconfig_file": "simulation_config.json"
    }
], indirect=True)
def test_neuromodulation_sims_neuron(create_tmp_simulation_config_file):
    import numpy.testing as npt
    from neurodamus import Neurodamus

    config_file = create_tmp_simulation_config_file
    nd = Neurodamus(config_file, disable_reports=True)
    nd.run()

    # compare spikes with refs
    spike_gids = np.array([1, 2, 2])  # 1-based
    timestamps = np.array([1.55, 2.025, 13.525])
    obtained_timestamps = nd._spike_vecs[0][0].as_numpy()
    obtained_spike_gids = nd._spike_vecs[0][1].as_numpy()
    npt.assert_allclose(spike_gids, obtained_spike_gids)
    npt.assert_allclose(timestamps, obtained_timestamps)


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "src_dir": str(SIM_DIR),
        "simconfig_file": "simulation_config.json",
        "extra_config" : {"target_simulator": "CORENEURON"}
    }
], indirect=True)
def test_neuromodulation_sims_coreneuron(create_tmp_simulation_config_file):
    from neurodamus import Neurodamus
    from neurodamus.replay import SpikeManager

    config_file = create_tmp_simulation_config_file
    nd = Neurodamus(config_file, disable_reports=True)
    nd.run()

    # compare spikes with refs
    spike_dat = Path(nd._run_conf.get("OutputRoot"))/nd._run_conf.get("SpikesFile")
    obtained_timestamps, obtained_spike_gids = SpikeManager._read_spikes_sonata(spike_dat, "All")
    spike_gids = np.array([1, 2, 2])  # 1-based
    timestamps = np.array([1.55, 2.025, 13.525])
    npt.assert_allclose(spike_gids, obtained_spike_gids)
    npt.assert_allclose(timestamps, obtained_timestamps)
