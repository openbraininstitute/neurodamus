import pytest
import tempfile
from pathlib import Path
from mpi4py import MPI


from tests.utils import defaultdict_to_standard_types
from ..conftest import PLATFORM_SYSTEM

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


@pytest.fixture(scope="session")
def tmp_folder():
    if rank == 0:
        path = tempfile.mkdtemp()
    else:
        path = None
    # Broadcast to all ranks
    path = comm.bcast(path, root=0)
    return path


@pytest.fixture(autouse=True)
def change_test_dir(monkeypatch, tmp_folder):
    """
    All tests in this file are using the same working directory, i.e tmp_folder
    Because test_dynamic_distribute requires memory_per_metype.json generated in the previous test
    """
    monkeypatch.chdir(tmp_folder)


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
    },
], indirect=True)
@pytest.mark.mpi(ranks=2)
def test_dry_run_memory_use(create_tmp_simulation_config_file, mpi_ranks):
    from neurodamus import Neurodamus

    nd = Neurodamus(create_tmp_simulation_config_file,  dry_run=True, num_target_ranks=2)
    nd.run()

    isMacOS = PLATFORM_SYSTEM == "Darwin"
    if rank == 0:
        assert (100.0 if isMacOS else 120.0) <= nd._dry_run_stats.base_memory <= (
            200.0 if isMacOS else 300.0)
        assert 0.4 <= nd._dry_run_stats.cell_memory_total <= 7.0
        assert 0.0 <= nd._dry_run_stats.synapse_memory_total <= 0.02
        expected_metypes_count = {
            'MTYPE1-ETYPE1': 2, 'MTYPE0-ETYPE0': 1,
            'MTYPE2-ETYPE2': 1, 'MTYPE0-ETYPE1': 1
        }
        assert nd._dry_run_stats.metype_counts == expected_metypes_count
        assert nd._dry_run_stats.suggested_nodes > 0


def is_subset(sub, main):
    return set(sub).issubset(set(main))


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
    },
], indirect=True)
@pytest.mark.mpi(ranks=2)
def test_dry_run_distribute_cells(create_tmp_simulation_config_file, mpi_ranks):
    from neurodamus import Neurodamus

    nd = Neurodamus(create_tmp_simulation_config_file,  dry_run=True, num_target_ranks=2)
    nd.run()

    rank_alloc = nd._dry_run_stats.import_allocation_stats(nd._dry_run_stats._ALLOCATION_FILENAME
                                                            + "_r2_c1.pkl.gz", 0)
    rank_allocation_standard = defaultdict_to_standard_types(rank_alloc)

    # Test allocation
    # RingA neuron 1 always in rank 0, neuron 2 always in rank 1
    # but neuron 3 can be in  either of the two
    if rank == 0:
        assert is_subset(rank_allocation_standard['RingA'][(0, 0)], [1, 3])
        assert rank_allocation_standard['RingB'][(0, 0)] == [1]
    elif rank == 1:
        assert is_subset(rank_allocation_standard['RingA'][(1, 0)], [2, 3])
        assert rank_allocation_standard['RingB'][(1, 0)] == [2]

    # Test redistribution
    rank_alloc, _, _ = nd._dry_run_stats.distribute_cells_with_validation(1, 1, None)
    rank_allocation_standard = defaultdict_to_standard_types(rank_alloc)
    expected_allocation = {
        'RingA': {(0, 0): [1, 2, 3]},
        'RingB': {(0, 0): [1, 2]}
    }
    assert rank_allocation_standard == expected_allocation

    rank_alloc = nd._dry_run_stats.import_allocation_stats(nd._dry_run_stats._ALLOCATION_FILENAME
                                                            + "_r1_c1.pkl.gz", 0, True)
    rank_allocation_standard = defaultdict_to_standard_types(rank_alloc)
    if rank == 0:
        expected_allocation = {
            'RingA': {(0, 0): [1, 2, 3]},
            'RingB': {(0, 0): [1, 2]}
        }
    elif rank == 1:
        expected_allocation = {
            'RingA': {},
            'RingB': {}
        }
    assert rank_allocation_standard == expected_allocation

    Path(("allocation_r1_c1.pkl.gz")).unlink(missing_ok=True)
    Path(("allocation_r2_c1.pkl.gz")).unlink(missing_ok=True)


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
    },
], indirect=True)
@pytest.mark.mpi(ranks=2)
def test_dry_run_dynamic_distribute(create_tmp_simulation_config_file, mpi_ranks):
    from neurodamus import Neurodamus

    nd = Neurodamus(create_tmp_simulation_config_file, dry_run=False, lb_mode="Memory",
                     num_target_ranks=2)
    nd.run()

    rank_alloc = nd._dry_run_stats.import_allocation_stats(nd._dry_run_stats._ALLOCATION_FILENAME
                                                            + "_r2_c1.pkl.gz", 0)
    rank_allocation_standard = defaultdict_to_standard_types(rank_alloc)

    # Test allocation
    # RingA neuron 1 always in rank 0, neuron 2 always in rank 1
    # but neuron 3 can be in  either of the two
    if rank == 0:
        assert is_subset(rank_allocation_standard['RingA'][(0, 0)], [1, 3])
    elif rank == 1:
        assert is_subset(rank_allocation_standard['RingA'][(1, 0)], [2, 3])
