#include <stdio.h>
#include <sys/time.h>
#include <mpi.h>

/* Program Parameters */
#define MAXN 2000  /* Max value of N */
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
    int my_rank, size;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int lims[3], i;

    printf("Started Process %d of %d\n", my_rank, (size-1));

    if (my_rank == 0)
    {
        //printf("Rank = %d\n", my_rank);

        /* Process program parameters */
        parameters(argc, argv);

        /* Initialize A and B */
        initialize_inputs();

        /* Print input matrices */
        print_inputs();


        for (i = 1; i < size; i++)
            {
                MPI_Send(A, (MAXN * MAXN), MPI_FLOAT, i, 0, MPI_COMM_WORLD);
                MPI_Send(B, MAXN, MPI_FLOAT, i, 1, MPI_COMM_WORLD);
                //MPI_Send(lims, 3, MPI_INT, i, 2, MPI_COMM_WORLD);
                printf("Sent data to process %d\n", i);
            }
        /* (Diagonal elements are not normalized to 1.  This is treated in back
         * substitution.)
         */
    }
    else
    {
        MPI_Recv(A, (MAXN * MAXN), MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(B, MAXN, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //MPI_Recv(lims, 3, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        print_inputs();
    }

    MPI_Finalize();

    return 0;
}
