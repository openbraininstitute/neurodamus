"""Test coreneuron simultion solver, directmode"""

from pathlib import Path

import pytest

from tests.utils import check_directory

from neurodamus import Neurodamus
from neurodamus.core.configuration import SimConfig


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "target_simulator": "CORENEURON",
        }
    }
], indirect=True)
def test_coreneuron_filemode(create_tmp_simulation_config_file):
    """Test the default coreneuron run
    1. coreneuron_input data folder, sim.conf, report.conf created and deleted after simulation
    2. populations_offset.dat created for reporting
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
            "node_set": "Mosaic"
        }
    }
], indirect=True)
def test_coreneuron_directmode(create_tmp_simulation_config_file):
    """Test coreneuron directmode
    1. empty coreneuron_input folder created and deleted after run
    2. sim.conf, report.conf, populations_offset.dat are created and deleted after run
    """
    from neurodamus import Neurodamus

    n = Neurodamus(create_tmp_simulation_config_file, coreneuron_direct_mode=True)
    coreneuron_data = Path(SimConfig.coreneuron_datadir)
    assert coreneuron_data.is_dir()
    assert not any(coreneuron_data.iterdir()), f"{coreneuron_data} should be empty."
    n.run()
    assert not coreneuron_data.exists()
