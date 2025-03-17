"""Test coreneuron simultion solver, directmode"""

from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from tests.utils import check_directory

from neurodamus import Neurodamus
from neurodamus.core.configuration import SimConfig


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "target_simulator": "CORENEURON",
            "Stimulus": {
                "module": "pulse",
                "input_type": "current_clamp",
                "delay": 5,
                "duration": 50,
                "node_set": "RingA",
                "represents_physical_electrode": True,
                "amp_start": 10,
                "width": 1,
                "frequency": 50
            }
        }
    }
], indirect=True)
def test_coreneuron_filemode(create_tmp_simulation_config_file):
    """Test the default coreneuron run
    1. coreneuron_input data folder, sim.conf, report.conf created and deleted after simulation
    2. populations_offset.dat created for reporting
    3. check the spikes on RingA from the pulse stimulus
    """
    n = Neurodamus(create_tmp_simulation_config_file)
    coreneuron_data = Path(SimConfig.coreneuron_datadir)
    check_directory(coreneuron_data)
    assert (Path(SimConfig.output_root) / "sim.conf").exists()
    assert (Path(SimConfig.output_root) / "report.conf").exists()
    assert (Path(SimConfig.output_root) / "populations_offset.dat").exists()
    n.run()
    assert not coreneuron_data.exists()
    assert not (Path(SimConfig.output_root) / "sim.conf").exists()
    assert not (Path(SimConfig.output_root) / "report.conf").exists()

    assert n._spike_populations[0][0] == "RingA"
    assert n._spike_populations[0][1] == 0
    ref_gids = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])  # 1-based
    ref_timestamps = np.array([5.1, 5.1, 5.1, 25.1, 25.1, 25.1, 45.1, 45.1, 45.1])
    npt.assert_allclose(n._spike_vecs[0][0].as_numpy(), ref_timestamps)
    npt.assert_allclose(n._spike_vecs[0][1].as_numpy(), ref_gids)

    assert n._spike_populations[1][0] == "RingB"
    assert n._spike_populations[1][1] == 1000
    assert n._spike_vecs[1][0].size() == 0  # no spike on RingB
    assert n._spike_vecs[1][1].size() == 0


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "target_simulator": "CORENEURON",
        }
    }
], indirect=True)
def test_coreneuron_disable_report(create_tmp_simulation_config_file):
    """Test coreneuron run disabling reporting
    1. report.conf, populations_offset.dat not created
    """
    n = Neurodamus(create_tmp_simulation_config_file, disable_reports=True)
    coreneuron_data = Path(SimConfig.coreneuron_datadir)
    check_directory(coreneuron_data)
    assert (Path(SimConfig.output_root) / "sim.conf").exists()
    assert not (Path(SimConfig.output_root) / "report.conf").exists()
    assert not (Path(SimConfig.output_root) / "populations_offset.dat").exists()
    n.run()
    assert not coreneuron_data.exists()
    assert not (Path(SimConfig.output_root) / "sim.conf").exists()


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "target_simulator": "CORENEURON",
        }
    }
], indirect=True)
def test_coreneuron_keep_build(create_tmp_simulation_config_file):
    """Test coreneuron run with CLI option keep_build
       coreneuron_input data folder, sim.conf, report.conf created and deleted after simulation
    """
    Neurodamus(create_tmp_simulation_config_file, keep_build=True).run()
    coreneuron_data = Path(SimConfig.coreneuron_datadir)
    check_directory(coreneuron_data)
    assert (Path(SimConfig.output_root) / "sim.conf").exists()
    assert (Path(SimConfig.output_root) / "report.conf").exists()
    assert (Path(SimConfig.output_root) / "populations_offset.dat").exists()


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "target_simulator": "CORENEURON",
            "node_set": "Mosaic",
            "inputs": {
                "Stimulus": {
                    "module": "pulse",
                    "input_type": "current_clamp",
                    "delay": 5,
                    "duration": 50,
                    "node_set": "RingA",
                    "represents_physical_electrode": True,
                    "amp_start": 10,
                    "width": 1,
                    "frequency": 50
                }
            }
        }
    }
], indirect=True)
def test_coreneuron_directmode(create_tmp_simulation_config_file):
    """Test coreneuron directmode
    1. empty coreneuron_input folder created and deleted after run
    2. sim.conf, report.conf, populations_offset.dat are created and deleted after run
    3. the same spike results as the file mode (test_coreneuron_filemode)
    """
    n = Neurodamus(create_tmp_simulation_config_file, coreneuron_direct_mode=True)
    coreneuron_data = Path(SimConfig.coreneuron_datadir)
    assert coreneuron_data.is_dir()
    assert not any(coreneuron_data.iterdir()), f"{coreneuron_data} should be empty."
    n.run()
    assert not coreneuron_data.exists()

    assert n._spike_populations[0][0] == "RingA"
    assert n._spike_populations[0][1] == 0
    ref_gids = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])  # 1-based
    ref_timestamps = np.array([5.1, 5.1, 5.1, 25.1, 25.1, 25.1, 45.1, 45.1, 45.1])
    npt.assert_allclose(n._spike_vecs[0][0].as_numpy(), ref_timestamps)
    npt.assert_allclose(n._spike_vecs[0][1].as_numpy(), ref_gids)

    assert n._spike_populations[1][0] == "RingB"
    assert n._spike_populations[1][1] == 1000
    assert n._spike_vecs[1][0].size() == 0  # no spike on RingB
    assert n._spike_vecs[1][1].size() == 0
