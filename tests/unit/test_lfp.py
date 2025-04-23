import pytest
from pathlib import Path
import h5py
import numpy as np

from .conftest import RINGTEST_DIR
LFP_FILE = RINGTEST_DIR / "lfp_file.h5"


def test_load_lfp_config():
    """
    Test that the 'load_lfp_config' function opens and loads correctly
    the LFP weights file and checks its format
    """
    from neurodamus.cell_distributor import LFPManager
    from neurodamus.core.configuration import ConfigurationError

    # Create an instance of the class
    lfp = LFPManager()
    pop_list = ["wrong_pop", "RingA", "RingB"]

    # Test loading LFP configuration from file
    lfp.load_lfp_config(LFP_FILE, pop_list)
    assert lfp._lfp_file
    assert isinstance(lfp._lfp_file, h5py.File)
    assert "/electrodes/RingA" in lfp._lfp_file
    assert "/RingA/node_ids" in lfp._lfp_file
    assert "/RingA/offsets" in lfp._lfp_file

    # Test loading LFP with no population found
    pop_list = ["wrong_pop"]
    with pytest.raises(ConfigurationError):
        lfp.load_lfp_config(LFP_FILE, pop_list)

    # Test loading LFP configuration from invalid file
    with pytest.raises(ConfigurationError):
        lfp.load_lfp_config("invalid_file.h5", pop_list)


def test_read_lfp_factors():
    """
    Test that the 'read_lfp_factors' function correctly extracts the LFP factors
    for the specified gid and section ids from the weights file
    """
    from neurodamus.cell_distributor import LFPManager
    # Create an instance of the class
    lfp = LFPManager()
    pop_list = ["RingA", "RingB"]
    lfp.load_lfp_config(LFP_FILE, pop_list)

    # Test the function with valid inputs for both populations
    result = lfp.read_lfp_factors(3, ("RingA", 0)).to_python()
    expected_result = [0.111, 0.112, 0.121, 0.122, 0.131, 0.132, 0.141, 0.142, 0.151, 0.152]
    assert result == expected_result, f'Expected {expected_result}, but got {result}'

    result = lfp.read_lfp_factors(1002, ("RingB", 1000)).to_python()
    expected_result = [
        0.064, 0.065, 0.066, 0.074, 0.075, 0.076, 0.084, 0.085, 0.086,
        0.094, 0.095, 0.096, 0.104, 0.105, 0.106
        ]
    assert result == expected_result

    # Test the function with invalid input
    # (non-existent gid)
    result_no_gid = lfp.read_lfp_factors(4, ("RingA", 0)).to_python()
    # (non-existent pop)
    result_no_pop = lfp.read_lfp_factors(1, ("WrongPop", 0)).to_python()
    # (wrong offset)
    result_wrong_offset = lfp.read_lfp_factors(1, ("RingA", 10)).to_python()
    expected_result = []
    assert result_no_gid == result_no_pop == result_wrong_offset == expected_result


def test_number_electrodes():
    """
    Test that the 'get_number_electrodes' function correctly extracts the number of
    electrodes in the weights file for a certain gid
    """
    from neurodamus.cell_distributor import LFPManager
    # Create an instance of the class
    lfp = LFPManager()
    pop_list = ["RingA", "RingB"]
    lfp.load_lfp_config(LFP_FILE, pop_list)

    # Test the function with valid input
    num_electrodes_1 = lfp.get_number_electrodes(1, ("RingA", 0))
    num_electrodes_2 = lfp.get_number_electrodes(2, ("RingA", 0))
    num_electrodes_3 = lfp.get_number_electrodes(3, ("RingA", 0))
    expected_num_electrodes = 2
    assert num_electrodes_1 == num_electrodes_2 == num_electrodes_3 == expected_num_electrodes

    num_electrodes_1001 = lfp.get_number_electrodes(1001, ("RingB", 1000))
    num_electrodes_1002 = lfp.get_number_electrodes(1002, ("RingB", 1000))
    expected_num_electrodes = 3
    assert num_electrodes_1001 == num_electrodes_1002 == expected_num_electrodes

    # Test the function with invalid input
    # (non-existent gid)
    result_no_gid = lfp.get_number_electrodes(4, ("RingA", 0))
    # (non-existent pop)
    result_no_pop = lfp.get_number_electrodes(1, ("WrongPop", 0))
    # (wrong offset)
    result_wrong_offset = lfp.get_number_electrodes(1, ("RingA", 10))
    expected_result = 0
    assert result_no_gid == result_no_pop == result_wrong_offset == expected_result


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "target_simulator": "CORENEURON",
            "run": {
                "electrodes_file": str(RINGTEST_DIR / "lfp_file.h5")
            },
            "reports": {
                "lfp": {
                    "type": "lfp",
                    "cells": "Mosaic",
                    "variable_name": "v",
                    "dt": 0.1,
                    "start_time": 0.0,
                    "end_time": 2.0
                }
            },
            "inputs": {
                "stimulus_pulse": {
                    "module": "pulse",
                    "input_type": "current_clamp",
                    "delay": 1,
                    "duration": 50,
                    "node_set": "RingA",
                    "represents_physical_electrode": True,
                    "amp_start": 10,
                    "width": 1,
                    "frequency": 50
                }
            }
        }
    },
], indirect=True)
@pytest.mark.forked
def test_lfp_reports(create_tmp_simulation_config_file):
    from neurodamus import Neurodamus
    from neurodamus.core import NeurodamusCore as Nd
    from neurodamus.core.configuration import SimConfig
    from tests.utils import record_lfp_report


    nd = Neurodamus(create_tmp_simulation_config_file)



    reports_conf = {name: conf for name, conf in SimConfig.reports.items()}
    recorders = {}
    for rep_name, rep_conf in reports_conf.items():
        rep_type = rep_conf["Type"]
        if rep_type == "lfp":
            recorders[rep_name] = record_lfp_report(rep_conf, nd._target_manager)

    Nd.finitialize()  # reinit for the recordings to be registered
    nd.run()

    assert len(recorders)>0
    for name, recorder in recorders.items():
        tvec = np.array(recorder[1])
        for gid, compartment_voltages in recorder[0].items():
            for i, voltages_vec in enumerate(compartment_voltages):
                voltages_vec = np.array(voltages_vec)
                assert len(voltages_vec) == len(tvec)

