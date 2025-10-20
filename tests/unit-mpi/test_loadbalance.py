import pytest
import numpy as np
import numpy.testing as npt
from tests import utils
from mpi4py import MPI
import tempfile

from neurodamus import Neurodamus

import numpy as np
from scipy.signal import find_peaks


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


@pytest.fixture(scope="function")
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
    Because for lb_modes WholeCell and MultiSplit requires distribution data files to be shared
    between all ranks
    """
    monkeypatch.chdir(tmp_folder)


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "inputs": {
                "Stimulus": {
                    "module": "pulse",
                    "input_type": "current_clamp",
                    "delay": 5,
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
@pytest.mark.parametrize("lb_mode", [
    "rr",
    "roundrobin",
    "wholecell",
    "loadbalance",
    "multisplit",
    "memory",
    "memory-cache"
])
@pytest.mark.mpi(ranks=2)
def test_load_balance_simulation(request, create_tmp_simulation_config_file, lb_mode,
                                 mpi_ranks):
    from neurodamus.core import NeuronWrapper as Nd

    if lb_mode == "memory-cache":
        request.getfixturevalue("copy_memory_files")       
        lb_mode = "memory"

    nd = Neurodamus(create_tmp_simulation_config_file, lb_mode=lb_mode)

    if rank == 0:
        cell_id = 1001
        manager = nd.circuits.get_node_manager("RingB")
        cell_ringB = manager.get_cell(cell_id)
        voltage_vec = Nd.Vector()
        voltage_vec.record(cell_ringB._cellref.soma[0](0.5)._ref_v)

    Nd.finitialize()
    nd.run()

    spike_gid_ref = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
    timestamps_ref = np.array([5.1, 5.1, 5.1, 25.1, 25.1, 25.1, 45.1, 45.1, 45.1])
    voltage_peaks_ref = np.array([92, 291])

    # Get spikes in RingA population for every rank
    ringA_spikes = nd._spike_vecs[0]
    timestamps = np.array(ringA_spikes[0])
    spike_gids = np.array(ringA_spikes[1])

    # Combine spike information from all ranks
    gathered_timestamps = comm.gather(timestamps, root=0)
    gathered_gids = comm.gather(spike_gids, root=0)
    if rank == 0:
        timestamps = np.concatenate(gathered_timestamps)
        spike_gids = np.concatenate(gathered_gids)

        combined = np.core.records.fromarrays([timestamps, spike_gids], names='timestamp, gid')
        sorted_combined = np.sort(combined, order=['timestamp', 'gid'])

        timestamps = sorted_combined.timestamp
        spike_gids = sorted_combined.gid

        # Check spikes in RingA population
        npt.assert_equal(spike_gid_ref, spike_gids)
        npt.assert_allclose(timestamps_ref, timestamps)

        # Check voltage variation in RingB population cell
        peaks_pos = find_peaks(voltage_vec, prominence=1)[0]
        np.testing.assert_allclose(peaks_pos, voltage_peaks_ref)
