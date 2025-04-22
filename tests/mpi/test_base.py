# tests/mpi/test_mpi_basic.py
import pytest
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def test_non_mpi():
    """Test non-MPI functionality."""
    assert True

@pytest.mark.mpi(min_size=2)
def test_mpi_send_recv():
    """Test basic MPI send/receive functionality."""
    assert size >= 2, "This test requires at least 2 MPI processes"

    if rank == 0:
        data = {"msg": "Hello from rank 0"}
        comm.send(data, dest=1, tag=0)
    elif rank == 1:
        data = comm.recv(source=0, tag=0)
        assert data["msg"] == "Hello from rank 0"




