# mpi_hello_world.py
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print(f"MPI vendor: {MPI.get_vendor()}")

print(f"Hello from rank {rank} out of {size}")

if size == 1:
    print("⚠️ Warning: MPI not properly initialized or only 1 process started")