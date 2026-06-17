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


def test_get_scaling_matrix():
    """
    Test that get_scaling_matrix correctly extracts the scaling factors
    as a 2D numpy array (n_compartments, n_electrodes).
    """
    from neurodamus.lfp_reader import LFPFileReader

    reader = LFPFileReader(str(LFP_FILE))

    # Test with valid inputs for both populations
    matrix = reader.get_scaling_matrix(2, ("RingA", 0))
    assert matrix is not None
    assert matrix.shape == (5, 2)  # 5 compartments, 2 electrodes
    expected_flat = np.array(
        [0.111, 0.112, 0.121, 0.122, 0.131, 0.132, 0.141, 0.142, 0.151, 0.152]
    )
    npt.assert_allclose(matrix.flatten(), expected_flat)

    matrix = reader.get_scaling_matrix(1001, ("RingB", 1000))
    assert matrix is not None
    assert matrix.shape == (5, 3)  # 5 compartments, 3 electrodes
    expected_flat = np.array([
        0.064, 0.065, 0.066, 0.074, 0.075, 0.076, 0.084, 0.085, 0.086,
        0.094, 0.095, 0.096, 0.104, 0.105, 0.106
    ])
    npt.assert_allclose(matrix.flatten(), expected_flat)

    # Test with invalid inputs — returns None
    assert reader.get_scaling_matrix(3, ("RingA", 0)) is None
    assert reader.get_scaling_matrix(0, ("WrongPop", 0)) is None
    assert reader.get_scaling_matrix(0, ("RingA", 10)) is None

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



def test_interleave_lfp_factors():
    """Test that _interleave_lfp_factors correctly interleaves per-report matrices."""
    from neurodamus.lfp_reader import LFPFileReader
    from neurodamus.core.coreneuron_configuration import CompartmentMapping

    reader_A = LFPFileReader(str(RINGTEST_DIR / "lfp_3elec_ringA.h5"))
    reader_B = LFPFileReader(str(RINGTEST_DIR / "lfp_2elec_ringA_cell0.h5"))

    # gid 0 is in both files
    pop_info = ("RingA", 0)
    all_factors, offsets = CompartmentMapping._interleave_lfp_factors(
        [reader_A, reader_B], 0, pop_info
    )

    # offsets: [0, 3, 5] — report A has 3 electrodes, report B has 2
    assert offsets == [0, 3, 5]

    # Verify interleaving: for each compartment, A's 3 values then B's 2 values
    matrix_A = reader_A.get_scaling_matrix(0, pop_info)
    matrix_B = reader_B.get_scaling_matrix(0, pop_info)
    expected = np.hstack([matrix_A, matrix_B]).flatten()
    npt.assert_allclose(np.array(all_factors.to_python()), expected)

    # gid 1 is only in A, not in B
    all_factors, offsets = CompartmentMapping._interleave_lfp_factors(
        [reader_A, reader_B], 1, pop_info
    )
    assert offsets == [0, 3, 3]  # B contributes 0 electrodes
    expected_A_only = matrix_A_gid1 = reader_A.get_scaling_matrix(1, pop_info).flatten()
    npt.assert_allclose(np.array(all_factors.to_python()), expected_A_only)

    reader_A.close()
    reader_B.close()
