import pytest
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


@pytest.mark.mpi(ranks=2)
def test_mpi_send_recv(mpi_ranks):
    """Test basic MPI send/receive functionality."""
    assert size == 2, "This test requires exactly 2 MPI processes"
    assert size == mpi_ranks, "MPI ranks do not match the expected number"

    if rank == 0:
        data = {"msg": "Hello from rank 0"}
        comm.send(data, dest=1, tag=0)
    elif rank == 1:
        data = comm.recv(source=0, tag=0)
        assert data["msg"] == "Hello from rank 0"


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "target_simulator": "NEURON",
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
            },
            "reports": {
                "soma_v": {
                    "type": "compartment",
                    "cells": "Mosaic",
                    "variable_name": "v",
                    "sections": "soma",
                    "dt": 0.1,
                    "start_time": 0.0,
                    "end_time": 18.0,
                },
                "compartment_i": {
                    "type": "compartment",
                    "cells": "Mosaic",
                    "variable_name": "i_membrane",
                    "sections": "all",
                    "dt": 1,
                    "start_time": 0.0,
                    "end_time": 40.0,
                },
            },
        }
    },
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "target_simulator": "CORENEURON",
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
            },
            "reports": {
                "soma_v": {
                    "type": "compartment",
                    "cells": "Mosaic",
                    "variable_name": "v",
                    "sections": "soma",
                    "dt": 0.1,
                    "start_time": 0.0,
                    "end_time": 18.0,
                },
                "compartment_i": {
                    "type": "compartment",
                    "cells": "Mosaic",
                    "variable_name": "i_membrane",
                    "sections": "all",
                    "dt": 1,
                    "start_time": 0.0,
                    "end_time": 40.0,
                },
            },
        }
    }
], indirect=True)
@pytest.mark.mpi(ranks=[2])
def test_neurodamus(create_tmp_simulation_config_file, mpi_ranks):
    """Test Neurodamus/neuron with and without MPI."""
    from neurodamus import Neurodamus
    from neurodamus.core import MPI
    # from neurodamus.core.configuration import SimConfig

    assert MPI.size == mpi_ranks == size
    assert MPI.rank == rank
    n = Neurodamus(create_tmp_simulation_config_file, disable_reports=True)
    local_gids_ref = [[1, 3], [2]]
    local_gids = n.circuits.get_node_manager("RingA").local_nodes.final_gids()
    import numpy.testing as npt
    npt.assert_allclose(local_gids, local_gids_ref[MPI.rank])
    n.run()
