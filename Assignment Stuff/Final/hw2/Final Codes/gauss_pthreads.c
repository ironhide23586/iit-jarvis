/* Gaussian elimination without pivoting.
 * Compile with "gcc gauss.c"
 */

/* ****** ADD YOUR CODE AT THE END OF THIS FILE. ******
 * You need not submit the provided code.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <time.h>

#include <pthread.h>

/* Program Parameters */
#define MAXN 2000  /* Max value of N */
int N;  /* Matrix size */

/* Matrices and vectors */
volatile float A[MAXN][MAXN], B[MAXN], X[MAXN];
/* A * X = B, solve for X */

/* junk */
#define randm() 4|2[uid]&3

/* Prototype */
void gauss_parallel();  /* The function you will provide.
		* It is this routine that is timed.
		* It is called only on the parent.
		*/

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

int main(int argc, char **argv) {
  /* Timing variables */
  struct timeval etstart, etstop;  /* Elapsed times using gettimeofday() */
  struct timezone tzdummy;
  clock_t etstart2, etstop2;  /* Elapsed times using times() */
  unsigned long long usecstart, usecstop;
  struct tms cputstart, cputstop;  /* CPU times for my processes */

  /* Process program parameters */
  parameters(argc, argv);

  /* Initialize A and B */
  initialize_inputs();

  /* Print input matrices */
  print_inputs();

  /* Start Clock */
  printf("\nStarting clock.\n");
  gettimeofday(&etstart, &tzdummy);
  etstart2 = times(&cputstart);

  /* Gaussian Elimination */
  gauss_parallel();

  /* Stop Clock */
  gettimeofday(&etstop, &tzdummy);
  etstop2 = times(&cputstop);
  printf("Stopped clock.\n");
  usecstart = (unsigned long long)etstart.tv_sec * 1000000 + etstart.tv_usec;
  usecstop = (unsigned long long)etstop.tv_sec * 1000000 + etstop.tv_usec;

  /* Display output */
  print_X();

  /* Display timing results */
  printf("\nElapsed time = %g ms.\n",
	 (float)(usecstop - usecstart)/(float)1000);

  printf("(CPU times are accurate to the nearest %g ms)\n",
	 1.0/(float)CLOCKS_PER_SEC * 1000.0);
  printf("My total CPU time for parent = %g ms.\n",
	 (float)( (cputstop.tms_utime + cputstop.tms_stime) -
		  (cputstart.tms_utime + cputstart.tms_stime) ) /
	 (float)CLOCKS_PER_SEC * 1000);
  printf("My system CPU time for parent = %g ms.\n",
	 (float)(cputstop.tms_stime - cputstart.tms_stime) /
	 (float)CLOCKS_PER_SEC * 1000);
  printf("My total CPU time for child processes = %g ms.\n",
	 (float)( (cputstop.tms_cutime + cputstop.tms_cstime) -
		  (cputstart.tms_cutime + cputstart.tms_cstime) ) /
	 (float)CLOCKS_PER_SEC * 1000);
      /* Contrary to the man pages, this appears not to include the parent */
  printf("--------------------------------------------\n");

  exit(0);
}

/* ------------------ Above Was Provided --------------------- */

/****** You will replace this routine with your own parallel version *******/
/* Provided global variables are MAXN, N, A[][], B[], and X[],
 * defined in the beginning of this code.  X[] is initialized to zeros.
 */
void *processRows(int *index) /*This part is executed in parallel by each thread.*/
{
    int startRow = *index, col, endRow = *(index + 1), norm = *(index + 2), row; /*Extracting the array index limits from input argument
										  *for each thread.
								   		  */
    float multiplier;
    for (row = startRow; row <= endRow; row++) /*Operating on the respective fraction of the Matrix corresponding to the thread.*/
    {
        multiplier = A[row][norm] / A[norm][norm];
        for (col = norm; col < N; col++)
        {
            A[row][col] -= A[norm][col] * multiplier;
        }
        B[row] -= B[norm] * multiplier;
    }
}

/* In this parallel implementation, the array is processed sequentially in iterations of the normalization row because of 
  * existence of Read-After-Write dependency in the outer-most loop (with the iterating variable 'norm').
  * However, each iteration pertaining to the normalization row has been parallelized. The number of rows beneath
  * the normalization row is divided between multiple threads whose number depends upon the number of logical processors
  * present on the target machine.
  * A dynamic array containing the row ranges for each thread is initialized. The starting memory address of the array blocks
  * pertaining to each thread is passed as an argument to each thread. The thread unpacks the values into variables
  * 'startRow', 'endRow' and 'norm' which are subsequently utilized for the computation process.
  */
void gauss_parallel() /*Function implementing a parallelized version of the Naive-Gauss elimination algorithm.*/
{
    int numCPU = sysconf( _SC_NPROCESSORS_ONLN ); /*Getting number of cores on Target Machine.
						   *Work on the matrix will be equally split across each core to ensure maximum performance.
						   */

    printf("Computing in Parallel on %d Logical cores\n", numCPU);

    if (N <= numCPU)	/*This handles the edge case when matrix size is smaller than number of
			 *cores on the machine.
			 */
        numCPU = N - 1;

    float f = (float) (N - 1) / numCPU;

    int blockSize = (unsigned int) f; /*Calculating number of rows each thread will be handling.*/
    if (f > blockSize)
	blockSize++;

    int norm, row, col;  /* Normalization row, and zeroing
			  * element row and col
			  */

    pthread_t *rowThreads;
    rowThreads = (pthread_t *)malloc(numCPU * sizeof(pthread_t)); /*Dynamically allocating memory for thread ids
								   *as the number of threads change according to
								   *number of logical cores on target machine.
								   */

    int nThreads, i, *indices;
    indices = (int *)malloc(3 * numCPU * sizeof(int)); /*This dynamic array contains sets of 3 values for each thread.
							*The first value is the index of the row from where the corresponding thread
							*shall begin processing its fraction of the input matrix.
							*The second value stores the respective ending row matrix.
							*The third value stores the index of the current normalization row.
							*These values are passed as arguments to each thread.
							*/

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

    /* Back substitution */
    for (row = N - 1; row >= 0; row--)
    {
        X[row] = B[row];
        for (col = N-1; col > row; col--)
        {
            X[row] -= A[row][col] * X[col];
        }
        X[row] /= A[row][row];
    }
}
