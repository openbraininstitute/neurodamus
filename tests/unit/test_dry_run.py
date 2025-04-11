import pytest
import numpy as np
import numpy.testing as npt
import unittest.mock
import tempfile
from pathlib import Path

from tests.utils import defaultdict_to_standard_types
from .conftest import RINGTEST_DIR, NGV_DIR
from ..conftest import PLATFORM_SYSTEM

TMP_FOLDER = tempfile.mkdtemp()


@pytest.fixture(autouse=True)
def change_test_dir(monkeypatch):
    """
    All tests in this file are using the same working directory, i.e TMP_FOLDER
    Because test_dynamic_distribute requires memory_per_metype.json generated in the previous test
    """
    monkeypatch.chdir(TMP_FOLDER)


@pytest.mark.forked
def test_dry_run_memory_use():
    from neurodamus import Neurodamus

    nd = Neurodamus(str(RINGTEST_DIR / "simulation_config.json"),  dry_run=True, num_target_ranks=2)

    nd.run()

    isMacOS = PLATFORM_SYSTEM == "Darwin"
    assert (45.0 if isMacOS else 80.0) <= nd._dry_run_stats.base_memory <= (
        75.0 if isMacOS else 120.0)
    assert 0.4 <= nd._dry_run_stats.cell_memory_total <= 6.0
    assert 0.0 <= nd._dry_run_stats.synapse_memory_total <= 0.02
    expected_metypes_count = {
        'MTYPE1-ETYPE1': 2, 'MTYPE0-ETYPE0': 1,
        'MTYPE2-ETYPE2': 1, 'MTYPE0-ETYPE1': 1
    }
    assert nd._dry_run_stats.metype_counts == expected_metypes_count
    assert nd._dry_run_stats.suggested_nodes > 0


@pytest.mark.forked
def test_dry_run_distribute_cells():
    from neurodamus import Neurodamus

    nd = Neurodamus(str(RINGTEST_DIR / "simulation_config.json"),  dry_run=True, num_target_ranks=2)
    nd.run()

    # Test allocation
    rank_alloc = nd._dry_run_stats.import_allocation_stats(nd._dry_run_stats._ALLOCATION_FILENAME +
                                                           "_r2_c1.pkl.gz", 0)
    rank_allocation_standard = defaultdict_to_standard_types(rank_alloc)
    expected_allocation = {
        'RingA': {(0, 0): [1]},
        'RingB': {(0, 0): [1]}
    }
    assert rank_allocation_standard == expected_allocation

    # Test redistribution
    rank_alloc, _, _ = nd._dry_run_stats.distribute_cells_with_validation(1, 1, None)
    rank_allocation_standard = defaultdict_to_standard_types(rank_alloc)
    expected_allocation = {
        'RingA': {(0, 0): [1, 2, 3]},
        'RingB': {(0, 0): [1, 2]}
    }
    assert rank_allocation_standard == expected_allocation

    rank_alloc = nd._dry_run_stats.import_allocation_stats(nd._dry_run_stats._ALLOCATION_FILENAME +
                                                           "_r1_c1.pkl.gz", 0, True)
    rank_allocation_standard = defaultdict_to_standard_types(rank_alloc)
    expected_allocation = {
        'RingA': {(0, 0): [1, 2, 3]},
        'RingB': {(0, 0): [1, 2]}
    }
    assert rank_allocation_standard == expected_allocation

    # Test reuse of cell_memory_use file
    rank_allocation, _, _ = nd._dry_run_stats.distribute_cells_with_validation(
        2, 1, nd._dry_run_stats._MEMORY_USAGE_PER_METYPE_FILENAME)
    rank_allocation_standard = defaultdict_to_standard_types(rank_allocation)
    expected_allocation = {
        'RingA': {
            (0, 0): [1],
            (1, 0): [2, 3]
        },
        'RingB': {
            (0, 0): [1],
            (1, 0): [2]
        }
    }
    assert rank_allocation_standard == expected_allocation

    Path(("allocation_r1_c1.pkl.gz")).unlink(missing_ok=True)
    Path(("allocation_r2_c1.pkl.gz")).unlink(missing_ok=True)


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
    },
], indirect=True)
@pytest.mark.forked
def test_dry_run_dynamic_distribute(create_tmp_simulation_config_file):
    from neurodamus import Neurodamus

    nd = Neurodamus(create_tmp_simulation_config_file, dry_run=False, lb_mode="Memory",
                     num_target_ranks=1)

    rank_alloc, _, _ = nd._dry_run_stats.distribute_cells_with_validation(2, 1)
    rank_allocation_standard = defaultdict_to_standard_types(rank_alloc)
    expected_allocation = {
        'RingA': {
            (0, 0): [1],
            (1, 0): [2, 3]
        }
    }
    assert rank_allocation_standard == expected_allocation


@pytest.mark.forked
def test_dry_run_ngv_fail():
    from neurodamus import Neurodamus

    with pytest.raises(Exception, match="Dry run not available for ngv circuit"):
        Neurodamus(str(NGV_DIR / "simulation_config.json"),  dry_run=True)


@pytest.mark.forked
def test_dry_run_distribution():
    """Test the dry_run_distribution function.
    This test makes sure that the distribution for dry runs
    returns the inner lists of the gid_metype_bundle as a single list
    with round robin distribution on the inner lists.
    """
    from neurodamus.io.cell_readers import dry_run_distribution

    # Sample of a typical gid_metype_bundle
    gid_metype_bundle = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10]]

    # Test with stride=1 (single rank)
    expected_output = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    npt.assert_equal(dry_run_distribution(gid_metype_bundle, stride=1), expected_output)

    # Test with stride=2 and stride_offset=0 (two ranks total, first rank)
    expected_output = np.array([1, 2, 3, 7, 8, 9])
    npt.assert_equal(dry_run_distribution(gid_metype_bundle, stride=2, stride_offset=0),
                     expected_output)

    # Test with stride=2 and stride_offset=1 (two ranks total, second rank)
    expected_output = np.array([4, 5, 6, 10])
    npt.assert_equal(dry_run_distribution(gid_metype_bundle, stride=2, stride_offset=1),
                     expected_output)


@pytest.mark.forked
def test_retrieve_unique_metypes():
    from neurodamus.io.cell_readers import _retrieve_unique_metypes

    # Define test inputs
    node_reader = DummyNodeReader()
    all_gids = [1, 2, 3, 4, 5]

    # Call the function
    with unittest.mock.patch('neurodamus.io.cell_readers.isinstance', return_value=True):
        result_list, metype_counts = _retrieve_unique_metypes(node_reader, all_gids)

    # Assertion checks
    assert isinstance(result_list, dict)
    assert all(isinstance(lst, np.ndarray) for lst in result_list.values())

    # Check the expected output based on the test inputs
    expected_result_dict = {'mtype1-emodel1': [1, 3, 5], 'mtype2-emodel2': [2, 4]}
    for metype, gids in result_list.items():
        npt.assert_equal(gids, expected_result_dict[metype])
    expected_metype_counts = {'mtype1-emodel1': 3, 'mtype2-emodel2': 2}
    assert metype_counts == expected_metype_counts


class DummyNodeReader:
    """ Fake dummy class to mock the NodeReader class
    for sonata readers.
    """
    def get_attribute(self, attr, selection):
        if attr == "etype":
            return ["emodel1", "emodel2", "emodel1", "emodel2", "emodel1"]
        elif attr == "mtype":
            return ["mtype1", "mtype2", "mtype1", "mtype2", "mtype1"]
        else:
            pytest.fail(f"Unsupported attribute: {attr}")


@pytest.fixture
def fixed_memory_measurements():
    with unittest.mock.patch('neurodamus.utils.memory.get_mem_usage_kb', return_value=100):
        with unittest.mock.patch('neurodamus.utils.memory.SynapseMemoryUsage.get_memory_usage',
                                 return_value=10):
            yield


def test_distribute_cells_multi_pop_multi_cycle(fixed_memory_measurements):
    from neurodamus.utils.memory import DryRunStats
    """
    Test that the distribute_cells_with_validation function works with multiple pops and cycles
    """
    stats = DryRunStats()

    # Mock data for testing
    stats.metype_memory = {
        'L4_PC-dSTUT': 50,
        'L4_MC-dSTUT': 30,
        'L4_MC-dNAC': 20,
        'L5_PC-dSTUT': 40
    }
    stats.metype_cell_syn_average = {
        'L4_PC-dSTUT': 5,
        'L4_MC-dSTUT': 3,
        'L4_MC-dNAC': 2,
        'L5_PC-dSTUT': 4
    }
    stats.pop_metype_gids = {
        'NodeA': {
            'L4_PC-dSTUT': [1, 2, 3],
            'L4_MC-dSTUT': [4, 5],
            'L4_MC-dNAC': [6],
            'L5_PC-dSTUT': [7, 8]
        },
        'NodeB': {
            'L4_PC-dSTUT': [9],
            'L4_MC-dSTUT': [10, 11],
            'L4_MC-dNAC': [12, 13],
            'L5_PC-dSTUT': [14]
        }
    }

    # Run the distribute_cells_with_validation function
    bucket_allocation, bucket_memory, metype_memory_usage = stats.distribute_cells_with_validation(
        num_ranks=2,
        cycles=2
    )
    rank_allocation_standard = defaultdict_to_standard_types(bucket_allocation)

    expected_allocation = {
        'NodeA': {
            (0, 0): [1, 6],
            (1, 0): [3, 8],
            (0, 1): [2, 7],
            (1, 1): [4, 5]
        },
        'NodeB': {
            (0, 0): [9],
            (1, 0): [11],
            (0, 1): [10, 14],
            (1, 1): [12, 13]
        }
    }
    expected_memory = {
        'NodeA': {
            (0, 0): 90,
            (1, 0): 110,
            (0, 1): 110,
            (1, 1): 80
        },
        'NodeB': {
            (0, 0): 60,
            (1, 0): 40,
            (0, 1): 90,
            (1, 1): 60
        }
    }

    # Assert that the results match the expected values
    assert rank_allocation_standard == expected_allocation
    assert bucket_memory == expected_memory
