import numpy as np
import pytest
from pathlib import Path

from tests.utils import read_sonata_spike_file

from ..conftest import SIM_DIR, USECASE3

@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "src_dir": str(SIM_DIR / "v5_sonata"),
        "simconfig_file": "simulation_config_mini.json",
        "extra_config": {
            "target_simulator": "CORENEURON",
        }
    }
], indirect=True)
def test_v5_sonata_multisteps(capsys, create_tmp_simulation_config_file):
    import numpy.testing as npt
    from neurodamus import Neurodamus

    config_file = create_tmp_simulation_config_file

    nd = Neurodamus(config_file, modelbuilding_steps=3)
    nd.run()

    # compare spikes with refs
    spike_gids = np.array([
        4, 2, 0
    ])  # 0-based
    timestamps = np.array([32.275, 37.2  , 39.15 ])
    spike_file = Path(nd._run_conf.get("OutputRoot")) / nd._run_conf.get("SpikesFile")
    obtained_timestamps, obtained_spike_gids = read_sonata_spike_file(spike_file)
    npt.assert_allclose(spike_gids, obtained_spike_gids)
    npt.assert_allclose(timestamps, obtained_timestamps)

    captured = capsys.readouterr()
    assert "MULTI-CYCLE RUN: 3 Cycles" in captured.out

@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "src_dir": str(USECASE3),
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
        40.9, 42.6, 45.2, 48.3, 49.9
    ])
    spike_file = Path(nd._run_conf.get("OutputRoot")) / nd._run_conf.get("SpikesFile")
    obtained_timestamps, obtained_spike_gids = read_sonata_spike_file(spike_file)
    npt.assert_allclose(spike_gids, obtained_spike_gids)
    npt.assert_allclose(timestamps, obtained_timestamps)
