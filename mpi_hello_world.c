#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);  // Initialize the MPI environment
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Get the rank of the process
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // Get the total number of processes

    printf("Hello from rank %d of %d\n", rank, size);

    MPI_Finalize();  // Finalize the MPI environment
    return 0;
}
