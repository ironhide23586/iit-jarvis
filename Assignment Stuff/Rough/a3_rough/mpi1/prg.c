#include <stdio.h>
#include <mpi.h>

int main(int argc, char** argv)
{
    int rank, size;

    char msg[] = "Wasssssaaaaap Ich hasse mein Leif Hundin!\n";
    char rcv[100];

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    printf("Process %d of size %d\n", rank, size);

    if (rank == 0)
    {
        MPI_Send(msg, 100, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
        MPI_Send(msg, 100, MPI_CHAR, 1, 1, MPI_COMM_WORLD);
        printf("Data sent!\n");
    }
    else if (rank == 1)
    {
        MPI_Recv(rcv, 100, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Received data: %s\n", rcv);
    }

    MPI_Finalize();

    return 0;
}
