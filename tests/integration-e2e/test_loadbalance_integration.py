"""Tests load balance."""
# Since a good deal of load balance tests are e2e we put all of them together in this group
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

SIM_DIR = Path(__file__).parent.parent.absolute() / "simulations"


def _read_complexity_file(base_dir, pattern, cx_pattern):
    # Construct the full pattern path
    full_pattern = Path(base_dir) / pattern / cx_pattern

    # Use glob to find files that match the pattern
    matching_files = Path(".").glob(str(full_pattern))

    # Read each matching file
    for file_path in matching_files:
        return file_path.read_text()

    return ""


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "src_dir": str(SIM_DIR / "usecase3"),
        "simconfig_file": "simulation_sonata.json",
        "extra_config": {
            "network": str(SIM_DIR / "usecase3" / "circuit_config_virtualpop.json"),
            "output": {
                "output_dir": "reporting"
            },
            "connection_overrides": [
                {
                    "name": "disconnect_virtual_proj_l4pc",
                    "source": "virtual_target",
                    "target": "l4pc",
                    "weight": 0.0
                },
                {
                    "name": "disconnect_l4pc_virtual_proj",
                    "source": "l4pc",
                    "target": "virtual_target",
                    "delay": 0.025,
                    "weight": 0.0
                }
            ]
        }
    }
], indirect=True)
def test_loadbal_integration(create_tmp_simulation_config_file):
    """Ensure given the right files are in the lbal dir, the correct situation is detected
    """
    from neurodamus import Neurodamus
    from neurodamus.core.configuration import GlobalConfig
    from neurodamus.replay import SpikeManager
    GlobalConfig.verbosity = 2

    # Add connection_overrides for the virtual population so the offsets are calculated before LB
    nd = Neurodamus(create_tmp_simulation_config_file, lb_mode="WholeCell")
    nd.run()

    # Check the complexity file
    base_dir = "sim_conf"
    pattern = "_loadbal_*.NodeA"  # Matches any hash and population
    cx_pattern = "cx_NodeA*#.dat"  # Matches any cx file with the pattern
    assert Path(base_dir).is_dir(), "Directory 'sim_conf' not found."
    cx_file = _read_complexity_file(base_dir, pattern, cx_pattern)
    lines = cx_file.splitlines()
    assert len(lines) == 5
    assert int(lines[1]) == 3, "Number of gids different than 3."
    # Gid should be without offset (1 instead of 1001)
    assert int(lines[3].split()[0]) == 1, "gid 1 not found."

    # check the spikes
    spike_dat = Path(nd._run_conf.get("OutputRoot")) / nd._run_conf.get("SpikesFile")

    timestamps_A, gids_A = SpikeManager._read_spikes_sonata(spike_dat, "NodeA")
    assert len(timestamps_A) == 21
    ref_times = np.array([0.2, 0.3, 0.3, 2.5, 3.4, 4.2, 5.5, 7.0, 7.4, 8.6, 13.8, 19.6, 25.7, 32.,
                          36.4, 38.5, 40.8, 42.6, 45.2, 48.3, 49.8])
    ref_gids = np.array([0, 1, 2, 0, 1, 2, 0, 0, 1, 2, 2, 2, 2, 2, 0, 2, 1, 0, 2, 0, 1])
    npt.assert_allclose(timestamps_A, ref_times)
    npt.assert_allclose(gids_A, ref_gids)
