import pytest
import h5py
import numpy as np
from pathlib import Path
from ..conftest import RINGTEST_DIR

SIM_DIR = Path(__file__).parent.parent.absolute() / "simulations"


@pytest.fixture
def test_weights_file(tmp_path):
    """
    Generates example weights file
    Returns the h5py.File obj and the file path
    """
    # Define populations and their GIDs
    populations = {
        "default": [42, 0, 4],
        "other_pop": [77777, 88888]
    }

    # Create a test HDF5 file with sample data
    test_file = h5py.File(tmp_path / "test_file.h5", 'w')

    for population, gids in populations.items():
        # Create population group
        population_group = test_file.create_group(population)

        # Create node_ids dataset
        population_group.create_dataset("node_ids", data=gids)

        # Create offsets dataset
        sec_ids_count = [2, 82, 140]
        total_segments = 224
        offsets = np.append(np.add.accumulate(sec_ids_count) - sec_ids_count, total_segments)
        population_group.create_dataset('offsets', data=offsets)

        # Create electrodes group and data dataset
        electrodes_group = test_file.create_group("electrodes/" + population)
        matrix = []
        # Fill first 2 rows
        matrix.append([0.1, 0.2])
        matrix.append([0.3, 0.4])

        # Fill the remaining rows up to total_segments
        incrementx = 0.0
        incrementy = 0.0
        for i in range(2, total_segments):
            value_x = 0.4 + incrementx
            value_y = 0.3 + incrementy
            matrix.append([value_x, value_y])
            incrementx += 0.001
            incrementy -= 0.0032
        electrodes_group.create_dataset("scaling_factors", dtype='f8', data=matrix)

    return test_file, str(tmp_path / "test_file.h5")


def test_load_lfp_config(test_weights_file):
    """
    Test that the 'load_lfp_config' function opens and loads correctly
    the LFP weights file and checks its format
    """
    from neurodamus.cell_distributor import LFPManager
    from neurodamus.core.configuration import ConfigurationError

    # Load the electrodes file
    _, lfp_weights_file = test_weights_file

    # Create an instance of the class
    lfp = LFPManager()
    pop_list = ["wrong_pop", "default"]

    # Test loading LFP configuration from file
    lfp.load_lfp_config(lfp_weights_file, pop_list)
    assert lfp._lfp_file
    assert isinstance(lfp._lfp_file, h5py.File)
    assert "/electrodes/default" in lfp._lfp_file
    assert "/default/node_ids" in lfp._lfp_file

    del lfp._lfp_file["default"]["node_ids"]
    with pytest.raises(ConfigurationError):
        lfp.load_lfp_config(lfp_weights_file, pop_list)

    # Test loading LFP configuration from invalid file
    lfp_weights_invalid_file = "./invalid_file.h5"
    with pytest.raises(ConfigurationError):
        lfp.load_lfp_config(lfp_weights_invalid_file, pop_list)


def test_read_lfp_factors(test_weights_file):
    """
    Test that the 'read_lfp_factors' function correctly extracts the LFP factors
    for the specified gid and section ids from the weights file
    """
    from neurodamus.cell_distributor import LFPManager
    # Create an instance of the class
    lfp = LFPManager()
    lfp._lfp_file, _ = test_weights_file
    # Test the function with valid input (node_id is 0 based, so expected 42 in the file)
    gid = 42
    result = lfp.read_lfp_factors(gid, ("default", 0)).to_python()
    expected_result = [0.1, 0.2, 0.3, 0.4]
    assert result == expected_result, f'Expected {expected_result}, but got {result}'

    # Test the function with invalid input (non-existent gid)
    gid = 419
    result = lfp.read_lfp_factors(gid, ("default", 0)).to_python()
    expected_result = []
    assert result == expected_result, f'Expected {expected_result}, but got {result}'


def test_number_electrodes(test_weights_file):
    """
    Test that the 'get_number_electrodes' function correctly extracts the number of
    electrodes in the weights file for a certain gid
    """
    from neurodamus.cell_distributor import LFPManager
    # Create an instance of the class
    lfp = LFPManager()
    lfp._lfp_file, _ = test_weights_file
    # Test the function with valid input
    gid = 0
    result = lfp.get_number_electrodes(gid, ("default", 0))
    expected_result = 2
    assert result == expected_result, f'Expected {expected_result}, but got {result}'

    # Test the function with invalid input (non-existent gid)
    gid = 419
    result = lfp.get_number_electrodes(gid, ("default", 0))
    expected_result = 0
    assert result == expected_result, f'Expected {expected_result}, but got {result}'


def _read_sonata_lfp_file(lfp_file):
    import libsonata
    report = libsonata.ElementReportReader(lfp_file)
    lfp_data = {}
    for pop_name in report.get_population_names():
        node_ids = report[pop_name].get_node_ids()
        data = report[pop_name].get()
        lfp_data[pop_name] = (node_ids, data)
    return lfp_data


def test_v5_sonata_lfp(test_weights_file, create_simulation_config_file_factory, tmp_path):
    pass
    import numpy.testing as npt
    import json
    from neurodamus import Neurodamus
    from neurodamus.core.coreneuron_configuration import CoreConfig

    _, lfp_weights_file = test_weights_file
    with open(str(SIM_DIR / "v5_sonata" / "simulation_config_mini.json")) as f:
        sim_config_data = json.load(f)
    params = {
        "extra_config": {
            "network": str(SIM_DIR / "v5_sonata" / "sub_mini5" / "circuit_config.json"),
            "target_simulator": "CORENEURON",
            "run": {"electrodes_file": lfp_weights_file},
            "reports": {
                "override_field": 1,
                "lfp": {
                    "type": "lfp",
                    "cells": "Mosaic",
                    "variable_name": "v",
                    "dt": 0.1,
                    "start_time": 0.0,
                    "end_time": 1.0
                }
            }
        }
    }
    config_file = create_simulation_config_file_factory(params, tmp_path, sim_config_data)

    nd = Neurodamus(config_file)
    nd.run()

    # compare results with refs
    t3_data = np.array([ 0.00028025, -0.0008968 ,  0.0014093 , -0.00450975])
    t7_data = np.array([ 0.00030916, -0.0009893 ,  0.00153002, -0.00489607])
    node_ids = np.array([0, 4])
    result_ids, result_data = _read_sonata_lfp_file(
        Path(CoreConfig.output_root) / "lfp.h5")["default"]

    npt.assert_allclose(result_data.data[3], t3_data, atol=1e-8)
    npt.assert_allclose(result_data.data[7], t7_data, atol=1e-8)
    npt.assert_allclose(result_ids, node_ids)


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "target_simulator": "CORENEURON",
            "run": {
                "electrodes_file": str(RINGTEST_DIR / "lfp_file.h5")
            },
            "reports": {
                "lfp_report": {
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
def test_ringcircuit_lfp(create_tmp_simulation_config_file):
    import numpy.testing as npt
    from neurodamus import Neurodamus
    from neurodamus.core.coreneuron_configuration import CoreConfig

    nd = Neurodamus(create_tmp_simulation_config_file)
    nd.run()

    # compare results with refs
    lfp_data = _read_sonata_lfp_file(Path(CoreConfig.output_root) / "lfp_report.h5")
    result_ids, result_data = lfp_data["RingA"]

    

    node_ids = np.array([0, 1, 2])
    t11_data = np.array([0.11541528, 0.12541528, 0.6154153, 0.62541527, 1.1154152, 1.1254153])
    t19_data = np.array([0.11362588, 0.12362587, 0.6136259, 0.6236259, 1.1136259, 1.1236259])

    npt.assert_allclose(result_ids, node_ids)
    npt.assert_allclose(result_data.data[11], t11_data)
    npt.assert_allclose(result_data.data[19], t19_data)

    result_ids, result_data = lfp_data["RingB"]

    node_ids = np.array([0, 1])
    t11_data = np.array(
        [6.4121537e-07, 6.4121537e-07, 6.4121537e-07, 6.4121537e-07, 6.4121537e-07, 6.4121537e-07])
    t19_data = np.array(
        [8.2200177e-07, 8.2200177e-07, 8.2200177e-07, 8.2200177e-07, 8.2200177e-07, 8.2200177e-07])

    npt.assert_allclose(result_ids, node_ids)
    npt.assert_allclose(result_data.data[11], t11_data)
    npt.assert_allclose(result_data.data[19], t19_data)
