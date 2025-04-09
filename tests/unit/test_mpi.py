import pytest
from neurodamus import Neurodamus

@pytest.mark.mpi(ranks=2)
@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "target_simulator": "NEURON",
        }
    },
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "target_simulator": "CORENEURON",
        }
    }
], indirect=True)
def test_mpi_rank_communication(create_tmp_simulation_config_file, mpi_ranks):
    nd = Neurodamus(create_tmp_simulation_config_file, disable_reports=True)
    nd.run()

    # print("AAAAAAA", nd._pc.nhost())
    # assert nd._pc.nhost() == 1
    # rank = comm.rank
    # size = comm.size

    # assert size == 2, f"Expected 2 ranks, got {size}"

    # if rank == 0:
    #     comm.send("Hello from 0", dest=1, tag=123)
    # elif rank == 1:
    #     msg = comm.recv(source=0, tag=123)
    #     assert msg == "Hello from 0"

# import pytest
# from mpi4py import MPI


# @pytest.mark.mpi(ranks=[2])

# def test_without_mpi(mpi_ranks, create_tmp_simulation_config_file, comm):

#     # """Simple passing test"""
#     
#     nd.run()
    # print(mpi_ranks, comm.rank)

    # rank = comm.rank
    # size = comm.size

    # assert size >= 2, f"Test requires at least 2 ranks, but got {size}"

    # if rank == 0:
    #     data = "Hello from rank 0"
    #     comm.send(data, dest=1, tag=42)
    # elif rank == 1:
    #     data = comm.recv(source=0, tag=42)
    #     assert data == "Hello from rank 0", f"Rank 1 received incorrect data: {data}"

# @pytest.mark.mpi(ranks=2)
# def test_with_mpi(mpi_ranks):  # pylint: disable=unused-argument
#     """Simple passing test"""
#     assert True  # replace with actual test code

# @pytest.mark.mpi(ranks=2)
# def test_one_failing_rank(mpi_ranks, comm):  # pylint: disable=unused-argument
#     """In case of just one process failing an assert, the test counts
#     as failed and the outputs are gathered from the processes."""

#     assert comm.rank <= 1

# from mpi4py import MPI
# from neurodamus import Neurodamus


# comm = MPI.COMM_WORLD

# @pytest.mark.parametrize("create_tmp_simulation_config_file", [
#     {
#         "simconfig_fixture": "ringtest_baseconfig",
#         "extra_config": {
#             "target_simulator": "NEURON",
#         }
#     },
#     {
#         "simconfig_fixture": "ringtest_baseconfig",
#         "extra_config": {
#             "target_simulator": "CORENEURON",
#         }
#     }
# ], indirect=True)
# @pytest.mark.mpi(minsize=2)
# def test_no_mpi():
#     from mpi4py import MPI
#     comm = MPI.COMM_WORLD
#     assert comm.size >= 2
    # nd = Neurodamus(create_tmp_simulation_config_file, disable_reports=True)
    # nd.run()

# @pytest.mark.parametrize("create_tmp_simulation_config_file", [
#     {
#         "simconfig_fixture": "ringtest_baseconfig",
#         "extra_config": {
#             "target_simulator": "NEURON",
#         }
#     },
#     {
#         "simconfig_fixture": "ringtest_baseconfig",
#         "extra_config": {
#             "target_simulator": "CORENEURON",
#         }
#     }
# ], indirect=True)
# @pytest.mark.mpi(minsize=2)
# def test_mpi(create_tmp_simulation_config_file):
#     pass
#     # nd = Neurodamus(create_tmp_simulation_config_file, disable_reports=True)
#     # nd.run()

# @pytest.mark.parametrize("create_tmp_simulation_config_file", [
#     {
#         "simconfig_fixture": "ringtest_baseconfig",
#         "extra_config": {
#             "target_simulator": "NEURON",
#         }
#     },
#     {
#         "simconfig_fixture": "ringtest_baseconfig",
#         "extra_config": {
#             "target_simulator": "CORENEURON",
#         }
#     }
# ], indirect=True)
# @pytest.mark.mpi(minsize=3)
# def test_mpi2(create_tmp_simulation_config_file):
#     pass
#     # nd = Neurodamus(create_tmp_simulation_config_file, disable_reports=True)
#     # nd.run()

# from neurodamus import Neurodamus

# @pytest.mark.parametrize("create_tmp_simulation_config_file", [
#     {
#         "simconfig_fixture": "ringtest_baseconfig",
#         "extra_config": {
#             "target_simulator": "NEURON",
#         }
#     }
# ], indirect=True)
# @pytest.mark.mpi(min_size=2)
# def test_mpi():
#     comm = MPI.COMM_WORLD
#     rank = comm.Get_rank()
#     size = comm.Get_size()
#     assert size >= 2
#     assert size <= 3
    # nd = Neurodamus(create_tmp_simulation_config_file, disable_reports=True)
    # nd.run()
