#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>

/* Program Parameters */
#define MAXN 2000  /* Max value of N *

/* Matrices and vectors */
volatile float A[MAXN][MAXN], B[MAXN], X[MAXN];
volatile float A2[MAXN][MAXN], B2[MAXN], X2[MAXN];
/* A * X = B, solve for X */

void gauss(int N) {
  int norm, row, col;  /* Normalization row, and zeroing
			* element row and col */
  float multiplier;

  printf("Computing Serially.\n");

  /* Gaussian elimination */
  for (norm = 0; norm < N - 1; norm++) {
    for (row = norm + 1; row < N; row++) {
      multiplier = A2[row][norm] / A2[norm][norm];
      for (col = norm; col < N; col++) {
	A2[row][col] -= A2[norm][col] * multiplier;
      }
      B2[row] -= B2[norm] * multiplier;
    }
  }


  /* (Diagonal elements are not normalized to 1.  This is treated in back
   * substitution.)
   */


  /* Back substitution */
  for (row = N - 1; row >= 0; row--) {
    X2[row] = B2[row];
    for (col = N-1; col > row; col--) {
      X2[row] -= A2[row][col] * X2[col];
    }
    X2[row] /= A2[row][row];
  }
}

void initialize_inputs(int N) {
  int row, col;

  printf("\nInitializing...\n");
  for (col = 0; col < N; col++) {
    for (row = 0; row < N; row++) {
      A[row][col] = (float)rand() / 32768.0;
      A2[row][col] = A[row][col];
    }
    B[col] = (float)rand() / 32768.0;
    B2[col] = B[col];
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
int parameters(int argc, char **argv) {
  int seed = 0;  /* Random seed */
  char uid[32]; /*User name */

  /* Read command-line arguments */
  srand(time_seed());  /* Randomize */
    int N;
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
  //printf("\nMatrix dimension N = %i.\n", N);
  return N;
}

/* Print input matrices */
void print_inputs(int N) {
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

void print_inputs2(int N) {
  int row, col;

  if (N < 10) {
    printf("\nA =\n\t");
    for (row = 0; row < N; row++) {
      for (col = 0; col < N; col++) {
	printf("%5.2f%s", A2[row][col], (col < N-1) ? ", " : ";\n\t");
      }
    }
    printf("\nB = [");
    for (col = 0; col < N; col++) {
      printf("%5.2f%s", B2[col], (col < N-1) ? "; " : "]\n");
    }
  }
}

void print_X(int N) {
  int row;

  if (N < 100) {
    printf("\nX = [");
    for (row = 0; row < N; row++) {
      printf("%5.2f%s", X[row], (row < N-1) ? "; " : "]\n");
    }
  }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int i,j,k;
    int map[MAXN];
    float c[MAXN], sum=0.0;
    float range=1.0;
    int N=parameters(argc, argv);  /* Matrix size */
    int rank, nprocs;
    double startTime, endTime;
    MPI_Status status;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   /* get current process id */
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs); /* get number of processes */

    //////////////////////////////////////////////////////////////////////////////////

    if (rank==0)
    {
        /* Initialize A and B */
        initialize_inputs(N);

        /* Print input matrices */
        print_inputs(N);

        int row, col;

        gauss(N);

        for(i=0;i<N;i++)
        {
            printf("\nx%d=%f\t",i,X2[i]);
        }
        printf("\n");
        startTime = MPI_Wtime();
    }

    //////////////////////////////////////////////////////////////////////////////////

    MPI_Bcast (&A[0][0],(MAXN*MAXN),MPI_FLOAT,0,MPI_COMM_WORLD);
    MPI_Bcast (B,N,MPI_FLOAT,0,MPI_COMM_WORLD);

    for(k=0;k<(N - 1);k++)
    {
        float f = (float) (N - k - 1) / nprocs;
        int blockSize = (unsigned int) f; /*Calculating number of rows each thread will be handling.*/
        if (f > blockSize)
            blockSize++;

        MPI_Bcast (&A[k][k],N-k,MPI_FLOAT,rank,MPI_COMM_WORLD);
        MPI_Bcast (&B[k],1,MPI_FLOAT,rank,MPI_COMM_WORLD);

        int startIndex = (rank*blockSize) + 1 + k;

        if (startIndex < N)
        {
            int finalIndex = (rank + 1)*blockSize + k;
            if (finalIndex >= N)
                finalIndex = N - 1;

            for (i = startIndex; i <= finalIndex; i++)
            {
                float multiplier = A[i][k] / A[k][k];
                for (j = k; j < N; j++) {
                    A[i][j] -= A[k][j] * multiplier;
                }
                B[i] -= B[k] * multiplier;
            }
        }
    }
    //////////////////////////////////////////////////////////////////////////////////

    if (rank==0)
    {
        int row, col;
        for (row = N - 1; row >= 0; row--) {
            X[row] = B[row];
        for (col = N-1; col > row; col--) {
            X[row] -= A[row][col] * X[col];
        }
        X[row] /= A[row][row];
    }

    //////////////////////////////////////////////////////////////////////////////////
    if (rank==0)
    {
        endTime = MPI_Wtime();
        printf("\nThe solution is:");
        for(i=0;i<N;i++)
        {
            printf("\nx%d=%f\t",i,X[i]);
        }

        printf("\n\nExecution time: %f\n", (endTime - startTime));
    }

    MPI_Finalize();
    return(0);
    }
}
