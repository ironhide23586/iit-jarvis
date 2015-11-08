#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>

/* Program Parameters */
#define MAXN 2000  /* Max value of N *

/* Matrices and vectors */
volatile float A[MAXN][MAXN], B[MAXN], X[MAXN];
/* A * X = B, solve for X */

/* Initialize A and B (and X to 0.0s) */
void initialize_inputs(int N) {
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

        startTime = MPI_Wtime();
    }

    //////////////////////////////////////////////////////////////////////////////////



    MPI_Bcast (&A[0][0],(MAXN*MAXN),MPI_FLOAT,0,MPI_COMM_WORLD);
    MPI_Bcast (B,N,MPI_FLOAT,0,MPI_COMM_WORLD);

    for(i=0; i<N; i++)
    {
        map[i]= i % nprocs;
    }

    for(k=0;k<N;k++)
    {
        //printf("******k = %d, RANK = %d*****************\n", k, rank);
        MPI_Bcast (&A[k][k],N-k,MPI_FLOAT,map[k],MPI_COMM_WORLD);
        //printf("******k = %d, RANK = %d*****************\n", k, rank);
        MPI_Bcast (&B[k],1,MPI_FLOAT,map[k],MPI_COMM_WORLD);
        //printf("******k = %d, RANK = %d*****************\n", k, rank);
        for(i= k+1; i<N; i++)
        {
            //printf("i = %d\n", i);
            if(map[i] == rank)
            {
                c[i]=A[i][k]/A[k][k];
            }
        }
        //printf("******k = %d, RANK = %d*****************\n", k, rank);
        for(i= k+1; i<N; i++)
        {
            if(map[i] == rank)
            {
                for(j=0;j<N;j++)
                {
                    A[i][j]=A[i][j]-( c[i]*A[k][j] );
                }
                B[i]=B[i]-( c[i]*B[k] );
            }
        }
        //printf("******k = %d, RANK = %d*****************\n", k, rank);
    }

    //////////////////////////////////////////////////////////////////////////////////

    if (rank==0)
    {
        X[N-1]=B[N-1]/A[N-1][N-1];
        for(i=N-2;i>=0;i--)
        {
            sum=0;

            for(j=i+1;j<N;j++)
            {
                sum=sum+A[i][j]*X[j];
            }

            X[i]=(B[i]-sum)/A[i][i];
        }
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
