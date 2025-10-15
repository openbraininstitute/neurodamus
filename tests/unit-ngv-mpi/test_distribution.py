import pytest
import numpy as np
import numpy.testing as npt
from mpi4py import MPI

from tests.conftest import NGV_DIR
from neurodamus import Neurodamus


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "src_dir": str(NGV_DIR),
        "simconfig_file": "simulation_config.json"
    }
], indirect=True)
@pytest.mark.mpi(ranks=2)
def test_distribution(create_tmp_simulation_config_file, mpi_ranks):
    n = Neurodamus(create_tmp_simulation_config_file)

    # Check allocation for RingA population
    local_cells_gids = n.circuits.get_node_manager("RingA").local_nodes.gids(raw_gids=False)
    if rank == 0:
        local_cells_gids_ref = [1000, 1002, 1004, 1006]
    elif rank == 1:
        local_cells_gids_ref = [1001, 1003, 1005]
    npt.assert_allclose(local_cells_gids, local_cells_gids_ref)

    # Check allocation for Astrocites
    local_astrocytes_gids = n.circuits.get_node_manager("AstrocyteA").local_nodes.gids(raw_gids=False)
    if rank == 0:
        local_astrocytes_gids_ref = [0, 2, 4]
    elif rank == 1:
        local_astrocytes_gids_ref = [1, 3]
    npt.assert_allclose(local_astrocytes_gids, local_astrocytes_gids_ref)

    n.run()

    # Check RingA cells spikes
    if rank == 0:
        spike_gid_ref = np.array([1000, 1002, 1004, 1006])
        timestamps_ref = np.array([2.075]*len(spike_gid_ref))
    elif rank == 1:
        spike_gid_ref = np.array([1001, 1003, 1005])
        timestamps_ref = np.array([2.075]*len(spike_gid_ref))
    ringA_spikes = n._spike_vecs[0]
    timestamps = np.array(ringA_spikes[0])
    spike_gids = np.array(ringA_spikes[1])
    npt.assert_equal(spike_gid_ref, spike_gids)
    npt.assert_allclose(timestamps_ref, timestamps)

    # Check AstrocytesA spikes
    if rank == 0:
        spike_gid_ref = np.array([0, 2])
        timestamps_ref = np.array([5.475, 7.675])
    elif rank == 1:
        spike_gid_ref = np.array([1, 3])
        timestamps_ref = np.array([6.725, 8.775])
    astrocyteA_spikes = n._spike_vecs[1]
    timestamps = np.array(astrocyteA_spikes[0])
    spike_gids = np.array(astrocyteA_spikes[1])
    npt.assert_equal(spike_gids, spike_gid_ref)
    npt.assert_allclose(timestamps, timestamps_ref)
