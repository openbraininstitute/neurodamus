import pytest
import h5py
import numpy as np
from pathlib import Path

SIM_DIR = Path(__file__).parent.parent.absolute() / "simulations"


@pytest.fixture
def test_file(tmp_path):
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


def test_load_lfp_config(test_file):
    """
    Test that the 'load_lfp_config' function opens and loads correctly
    the LFP weights file and checks its format
    """
    from neurodamus.cell_distributor import LFPManager
    from neurodamus.core.configuration import ConfigurationError

    # Load the electrodes file
    _, lfp_weights_file = test_file

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


def test_read_lfp_factors(test_file):
    """
    Test that the 'read_lfp_factors' function correctly extracts the LFP factors
    for the specified gid and section ids from the weights file
    """
    from neurodamus.cell_distributor import LFPManager
    # Create an instance of the class
    lfp = LFPManager()
    lfp._lfp_file, _ = test_file
    # Test the function with valid input (node_id is 0 based, so expected 42 in the file)
    gid = 43
    result = lfp.read_lfp_factors(gid, ("default", 0)).to_python()
    expected_result = [0.1, 0.2, 0.3, 0.4]
    assert result == expected_result, f'Expected {expected_result}, but got {result}'

    # Test the function with invalid input (non-existent gid)
    gid = 420
    result = lfp.read_lfp_factors(gid, ("default", 0)).to_python()
    expected_result = []
    assert result == expected_result, f'Expected {expected_result}, but got {result}'


def test_number_electrodes(test_file):
    """
    Test that the 'get_number_electrodes' function correctly extracts the number of
    electrodes in the weights file for a certain gid
    """
    from neurodamus.cell_distributor import LFPManager
    # Create an instance of the class
    lfp = LFPManager()
    lfp._lfp_file, _ = test_file
    # Test the function with valid input
    gid = 1
    result = lfp.get_number_electrodes(gid, ("default", 0))
    expected_result = 2
    assert result == expected_result, f'Expected {expected_result}, but got {result}'

    # Test the function with invalid input (non-existent gid)
    gid = 420
    result = lfp.get_number_electrodes(gid, ("default", 0))
    expected_result = 0
    assert result == expected_result, f'Expected {expected_result}, but got {result}'


def _read_sonata_lfp_file(lfp_file):
    import libsonata
    report = libsonata.ElementReportReader(lfp_file)
    pop_name = report.get_population_names()[0]
    node_ids = report[pop_name].get_node_ids()
    data = report[pop_name].get()
    return node_ids, data


def _create_lfp_config(original_config_path, lfp_file, base_dir: Path):
    """
    Create a modified lfp configuration file in a temporary directory.
    """
    import json
    # Read the original config file
    with open(original_config_path, 'r') as f:
        config = json.load(f)

    # Update the network path in the config
    config["network"] = str(SIM_DIR / "v5_sonata" / "sub_mini5" / "circuit_config.json")

    # Modify the necessary fields
    config["target_simulator"] = "CORENEURON"
    config["run"]["electrodes_file"] = str(lfp_file)
    config["reports"] = config.get("reports", {})
    config["reports"]["lfp"] = {
        "type": "lfp",
        "cells": "Mosaic",
        "variable_name": "v",
        "dt": 0.1,
        "start_time": 0.0,
        "end_time": 1.0
    }

    # Write the modified configuration to a temporary file
    temp_config_path = base_dir / "lfp_config.json"
    with open(temp_config_path, "w") as f:
        json.dump(config, f, indent=4)

    # Create output directory
    output_dir = base_dir / config["output"]["output_dir"]

    return str(temp_config_path), str(output_dir)


def test_v5_sonata_lfp(test_file, tmp_path):
    import numpy.testing as npt
    from neurodamus import Neurodamus

    config_file = SIM_DIR / "v5_sonata" / "simulation_config_mini.json"
    _, lfp_weights_file = test_file
    temp_config_path, output_dir = _create_lfp_config(config_file, lfp_weights_file, tmp_path)

    nd = Neurodamus(temp_config_path, output_path=output_dir)
    nd.run()

    # compare results with refs
    # t3_data = np.array([0.00027065672, -0.00086610153, 0.0014563566, -0.0046603414])
    # t7_data = np.array([0.00029265403, -0.0009364929, 0.001548515, -0.004955248])
    node_ids = np.array([0, 4])
    result_ids, result_data = _read_sonata_lfp_file(Path(output_dir) / "lfp.h5")

    # TODO: reenable after: https://github.com/openbraininstitute/neurodamus/issues/3
    # is solved
    # npt.assert_allclose(result_data.data[3], t3_data)
    # npt.assert_allclose(result_data.data[7], t7_data)
    npt.assert_allclose(result_ids, node_ids)
