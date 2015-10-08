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
/* In this parallel implementation, the array is processed sequentially in iterations of the normalization row because of 
  * existence of Read-After-Write dependency in the outer-most loop (with the iterating variable 'norm').
  * However, each iteration pertaining to the normalization row has been parallelized. The number of rows beneath
  * the normalization row is divided between multiple threads whose number depends upon the number of logical processors
  * present on the target machine.
  * Furthermore, OpenMP's feature to easily parallelize rows has been employed here to parallelize the
  * loop processing each row inside a thread hence implementing in essence, a sort of branched parallelism.
  */
void gauss_parallel()
{
    int numCPU = sysconf( _SC_NPROCESSORS_ONLN ); /*Getting number of Logical cores on target machine.*/
    printf("Computing in Parallel on %d Logical cores\n", numCPU);

    int blockSize, vThreads, norm, row, col, i;
    float multiplier, f;
    /* Gaussian elimination */
    for (norm = 0; norm < N - 1; norm++)
    {
	f = (float) (N - norm - 1) / numCPU;
    	blockSize = (unsigned int) f; /*Calculating number of rows each thread will be handling.*/
    	if (f > blockSize)
            blockSize++;

        vThreads = blockSize * numCPU; /*Adjusting number of threads for edge cases to eliminate empty threads*/
        if (vThreads > (N-1))
            numCPU--;

        int tid; /*Variable to contain thread id pertaining to each thread.*/
        #pragma omp parallel num_threads(numCPU) firstprivate(norm, blockSize) private(tid, multiplier) /*Launching threads in parallel
                                                                                                         *to carry out implementation of the
                                                                                                         *Naive-Gauss elimination algorithm.
                                                                                                         *The number of thread launched is
                                                                                                         *equal to the number of logical
                                                                                                         *cores on the target machine.
                                                                                                         *Each  thread gets its own copy of the
                                                                                                         *norm variable which changes in the
                                                                                                         *outer loop.
                                                                                                         *Also, the blockSize variable i.e. the
                                                                                                         *number of arrays each thread handles also
                                                                                                         changes in each outer loop iteration and each
                                                                                                         *thread also gets its copy of it.
                                                                                                         */
        {
            tid = omp_get_thread_num(); /*Getting thread id.*/
            int startRow = norm + tid * blockSize + 1, col; /*Using thread id to calculate starting index of row
                                                             *which the thread handles.
                                                             */
            int endRow = startRow + blockSize - 1; /*Using thread id to calculate ending index of row
                                                    *which the thread handles.
                                                    */
            if (endRow >= N) /*Handling edge case when the final row pertaining to the thread is out of index range.*/
                endRow = N - 1;

            #pragma omp parallel for schedule(guided) /*Main part of each concurrent thread. This carries out the
                                                       *Naive-Gauss Elimination for the set of rows pertaining to the thread.
                                                       */
            for (row = startRow; row <= endRow; row++) /*Iterating through each row in row set.*/
            {
                multiplier = A[row][norm] / A[norm][norm];
                for (col = norm; col < N; col++)
                {
                    A[row][col] -= A[norm][col] * multiplier;
                }
                B[row] -= B[norm] * multiplier;
            }
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
