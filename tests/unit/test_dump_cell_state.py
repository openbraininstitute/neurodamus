"""
Test that the dump_cell_state option works correctly for NEURON and CORENEURON.
The dump_cell_state option allows the user to specify which gid cell states should be dumped.

Neuron allows to dump multiple cell states at once, and the user can specify
lists and ranges of cell states. CoreNeuron only allows to dump a single cell state
at once, and the user can only specify a single gid.
If multiple cell states are specified, the first one is used.
"""

import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "target_simulator": "NEURON",
            "run": {
                "tstop": 10,
            }
        }
    }
], indirect=True)
def test_neuron_nodump(create_tmp_simulation_config_file):
    """
    Test NEURON simulation without dumping cell state files.
    """
    command = ["neurodamus", "simulation_config.json"]
    subprocess.run(command, check=True, capture_output=True)
    assert not any(file.suffix == '.nrndat' for file in Path("output").glob("*.nrndat"))


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "target_simulator": "NEURON",
            "run": {
                "tstop": 10,
            }
        }
    }
], indirect=True)
def test_neuron_single(create_tmp_simulation_config_file):
    """
    Test NEURON simulation with a single cell state dump at t=0 and t=10.
    """
    command = ["neurodamus", "simulation_config.json", "--dump-cell-state=0"]
    subprocess.run(command, check=True, capture_output=True)
    for t in [0, 10]:
        assert Path(f"output/0_py_Neuron_t{t:0.1f}.nrndat").exists()


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "target_simulator": "NEURON",
            "run": {
                "tstop": 10,
            }
        }
    }
], indirect=True)
def test_neuron_list(create_tmp_simulation_config_file):
    """
    Test NEURON simulation with a list of cell state dumps (0, 1, 2).
    """
    command = ["neurodamus", "simulation_config.json", "--dump-cell-state=0,1,2,2,1"]
    subprocess.run(command, check=True, capture_output=True)
    for t in [0, 10]:
        for i in range(3):
            assert Path(f"output/{i}_py_Neuron_t{t:0.1f}.nrndat").exists()


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "target_simulator": "NEURON",
            "run": {
                "tstop": 10,
            }
        }
    }
], indirect=True)
def test_neuron_range(create_tmp_simulation_config_file):
    """
    Test NEURON simulation with a range of cell state dumps (0-2).
    """
    command = ["neurodamus", "simulation_config.json", "--dump-cell-state=0-2"]
    subprocess.run(command, check=True, capture_output=True)
    for t in [0, 10]:
        for i in range(3):
            assert Path(f"output/{i}_py_Neuron_t{t:0.1f}.nrndat").exists()


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "target_simulator": "NEURON",
            "run": {
                "tstop": 10,
            }
        }
    }
], indirect=True)
def test_neuron_list_and_range(create_tmp_simulation_config_file):
    """
    Test NEURON simulation with a combination of list and range of cell state dumps.
    """
    command = ["neurodamus", "simulation_config.json", "--dump-cell-state=0-2,1000,1001"]
    subprocess.run(command, check=True, capture_output=True)
    for t in [0, 10]:
        for i in [0, 1, 2, 1000, 1001]:
            assert Path(f"output/{i}_py_Neuron_t{t:0.1f}.nrndat").exists()


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "target_simulator": "CORENEURON",
            "run": {
                "tstop": 10,
            }
        }
    }
], indirect=True)
def test_coreneuron(create_tmp_simulation_config_file):
    """
    Test CORENEURON simulation with a single cell state dump at t=0 and t=10.
    """
    command = ["neurodamus", "simulation_config.json", "--dump-cell-state=2,0"]
    subprocess.run(command, check=True, capture_output=True)
    assert Path("output/2_cpu_init.corenrn").exists()
    assert Path("output/2_cpu_t0.000000.corenrn").exists()
    assert Path("output/2_cpu_t10.000000.corenrn").exists()
