import numpy.testing as npt
import numpy as np
import pytest
import h5py

from ..conftest import RINGTEST_DIR
LFP_FILE = RINGTEST_DIR / "lfp_file.h5"


def test_lfp_file_reader_open():
    """
    Test that LFPFileReader opens and validates the electrodes file structure.
    """
    from neurodamus.lfp_reader import LFPFileReader
    from neurodamus.core.configuration import ConfigurationError

    # Valid file opens without error
    reader = LFPFileReader(str(LFP_FILE))
    assert reader._file
    assert isinstance(reader._file, h5py.File)
    assert "/electrodes/RingA" in reader._file
    assert "/RingA/node_ids" in reader._file
    assert "/RingA/offsets" in reader._file
    reader.close()

    # Invalid file raises ConfigurationError
    with pytest.raises(ConfigurationError):
        LFPFileReader("invalid_file.h5")


def test_read_lfp_factors():
    """
    Test that get_factors correctly extracts the LFP factors
    for the specified gid and section ids from the weights file.
    """
    from neurodamus.lfp_reader import LFPFileReader

    reader = LFPFileReader(str(LFP_FILE))

    # Test with valid inputs for both populations
    result = reader.get_factors(2, ("RingA", 0)).to_python()
    expected_result = np.array(
        [0.111, 0.112, 0.121, 0.122, 0.131, 0.132, 0.141, 0.142, 0.151, 0.152]
    )
    npt.assert_allclose(result, expected_result)

    result = reader.get_factors(1001, ("RingB", 1000)).to_python()
    expected_result = np.array([
        0.064, 0.065, 0.066, 0.074, 0.075, 0.076, 0.084, 0.085, 0.086,
        0.094, 0.095, 0.096, 0.104, 0.105, 0.106
    ])
    npt.assert_allclose(result, expected_result)

    # Test with invalid inputs
    result_no_gid = reader.get_factors(3, ("RingA", 0)).to_python()
    result_no_pop = reader.get_factors(0, ("WrongPop", 0)).to_python()
    result_wrong_offset = reader.get_factors(0, ("RingA", 10)).to_python()
    expected_result = []
    assert result_no_gid == result_no_pop == result_wrong_offset == expected_result

    reader.close()


def test_number_electrodes():
    """
    Test that get_number_electrodes correctly extracts the number of
    electrodes in the weights file for a certain gid.
    """
    from neurodamus.lfp_reader import LFPFileReader

    reader = LFPFileReader(str(LFP_FILE))

    # Test with valid input
    num_electrodes_1 = reader.get_number_electrodes(0, ("RingA", 0))
    num_electrodes_2 = reader.get_number_electrodes(1, ("RingA", 0))
    num_electrodes_3 = reader.get_number_electrodes(2, ("RingA", 0))
    expected_num_electrodes = 2
    assert num_electrodes_1 == num_electrodes_2 == num_electrodes_3 == expected_num_electrodes

    num_electrodes_1001 = reader.get_number_electrodes(1000, ("RingB", 1000))
    num_electrodes_1002 = reader.get_number_electrodes(1001, ("RingB", 1000))
    expected_num_electrodes = 3
    assert num_electrodes_1001 == num_electrodes_1002 == expected_num_electrodes

    # Test with invalid inputs
    result_no_gid = reader.get_number_electrodes(3, ("RingA", 0))
    result_no_pop = reader.get_number_electrodes(0, ("WrongPop", 0))
    result_wrong_offset = reader.get_number_electrodes(0, ("RingA", 10))
    expected_result = 0
    assert result_no_gid == result_no_pop == result_wrong_offset == expected_result

    reader.close()
