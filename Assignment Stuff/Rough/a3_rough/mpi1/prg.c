#include <stdio.h>
#include <sys/time.h>
#include <mpi.h>

#define MAXN 2000

int N;  /* Matrix size */

/* Matrices and vectors */
volatile float A[MAXN][MAXN], B[MAXN], X[MAXN];
/* A * X = B, solve for X */

/* Initialize A and B (and X to 0.0s) */
void initialize_inputs() {
  int row, col;

  printf("\nInitializing...\n");
  for (col = 0; col < N; col++) {
    for (row = 0; row < N; row++) {
      A[row][col] = (float)rand() / 32768.0;
    }
    B[col] = (float)rand() / 32768.0;
    X[col] = 0.0;
  }

}

/* returns a seed for srand based on the time */
unsigned int time_seed() {
  struct timeval t;
  struct timezone tzdummy;

  gettimeofday(&t, &tzdummy);
  return (unsigned int)(t.tv_usec);
}

/* Set the program parameters from the command-line arguments */
void parameters(int argc, char **argv) {
  int seed = 0;  /* Random seed */
  char uid[32]; /*User name */

  /* Read command-line arguments */
  srand(time_seed());  /* Randomize */

  if (argc == 3) {
    seed = atoi(argv[2]);
    srand(seed);
    printf("Random seed = %i\n", seed);
  }
  if (argc >= 2) {
    N = atoi(argv[1]);
    if (N < 1 || N > MAXN) {
      printf("N = %i is out of range.\n", N);
      exit(0);
    }
  }
  else {
    printf("Usage: %s <matrix_dimension> [random seed]\n",
           argv[0]);
    exit(0);
  }

  /* Print parameters */
  printf("\nMatrix dimension N = %i.\n", N);
}

/* Print input matrices */
void print_inputs() {
  int row, col;

  if (N < 10) {
    printf("\nA =\n\t");
    for (row = 0; row < N; row++) {
      for (col = 0; col < N; col++) {
	printf("%5.2f%s", A[row][col], (col < N-1) ? ", " : ";\n\t");
      }
    }
    printf("\nB = [");
    for (col = 0; col < N; col++) {
      printf("%5.2f%s", B[col], (col < N-1) ? "; " : "]\n");
    }
  }
}

void print_X() {
  int row;

  if (N < 100) {
    printf("\nX = [");
    for (row = 0; row < N; row++) {
      printf("%5.2f%s", X[row], (row < N-1) ? "; " : "]\n");
    }
  }
}

int main(int argc, char** argv)
{
    int my_rank, size, i;

    //char msg0[] = "Wasssssaaaaap Ich hasse mein Leif Hundin!\n";
    //char msg1[] = "Lol I hate my life\n";
    //char rcv0[100], rcv1[100];

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    printf("Started Process %d of %d\n", my_rank, (size - 1));

    if (my_rank == 0)
    {
        printf("Computing in Parallel on %d Processes\n", size);

        /* Process program parameters */
        parameters(argc, argv);

        /* Initialize A and B */
        initialize_inputs();

        /* Print input matrices */
        print_inputs();

        /* Gaussian elimination */
        for (norm = 0; norm < N - 1; norm++) /*Proceeding sequentially on each norm row because of
                          *Read-After-Write dependence between each norm variable iteration.
                          */
        {
            i = 0;
            for (row = norm + 1; row < N; row += blockSize) /*Putting values in the 'inidices' dynamic array described above.
                                 *Note that this loop increments with a step size equal to the blockSize value
                                 *which is the number of rows each thread will be handling.
                                 */
            {
                indices[3 * i] = row; /*First value storing the starting row index.*/

                if ((row + blockSize - 1) < N) /*Second value stores the ending row index.*/
                    indices[3 * i + 1] = row + blockSize - 1;
                else
                    indices[3 * i + 1] = N - 1;

                indices[3 * i + 2] = norm; /*Third value stores value of current normalization row index.*/
                i++;
            }

        numCPU = i; /*Ensures that number of threads launched is equal to the number of proceesing lbocks made.*/

            for (i = 0; i < numCPU; i++)
            {
                pthread_create(rowThreads + i, NULL, processRows, (indices + 3 * i)); /*Launching each thread to operate on different parts of the array*/
            }

            for (i = 0; i < numCPU; i++)
            {
                pthread_join(*(rowThreads + i), NULL); /*Consolidating all threads*/
            }
        }
        /* (Diagonal elements are not normalized to 1.  This is treated in back
         * substitution.)
         */

        for (i = 1; i < size; i++)
        {
            MPI_Send(A, (MAXN*MAXN), MPI_FLOAT, i, 0, MPI_COMM_WORLD);
            MPI_Send(B, MAXN, MPI_FLOAT, i, 1, MPI_COMM_WORLD);
            MPI_Send(&N, 1, MPI_INT, i, 2, MPI_COMM_WORLD);
            printf("Data sent to processor %d!\n", i);
        }
    }
    else
    {
        MPI_Recv(A, (MAXN*MAXN), MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(B, MAXN, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&N, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Received size of data (Value of N) = %d\n", N);
        printf("Received data with tag 0 & 1\n");
        print_inputs();
    }

    MPI_Finalize();

    return 0;
}
