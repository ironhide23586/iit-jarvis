/* Matrix normalization.
 * Compile with "gcc matrixNorm.c"
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
#define MAXN 8000  /* Max value of N */
int N;  /* Matrix size */

/* Matrices */
volatile float A[MAXN][MAXN], B[MAXN][MAXN];

/* junk */
#define randm() 4|2[uid]&3

/* Prototype */
void matrixNorm_GPU();

__global__ void blank_function();

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

/* Initialize A and B*/
void initialize_inputs() {
  int row, col;

  printf("\nInitializing...\n");
  for (col = 0; col < N; col++) {
    for (row = 0; row < N; row++) {
      A[row][col] = (float)rand() / 32768.0;
      B[row][col] = 0.0;
    }
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
  }
}

void print_B() {
    int row, col;

    if (N < 10) {
        printf("\nB =\n\t");
        for (row = 0; row < N; row++) {
            for (col = 0; col < N; col++) {
                printf("%1.10f%s", B[row][col], (col < N-1) ? ", " : ";\n\t");
            }
        }
    }
}

int main(int argc, char **argv) {

  blank_function<<<1, 1>>>(); /* Blank Function (defined below) having no content which is simply used to initialize CUDA environment as it is implicitly initialized on first call and this delay is not caused by parallelized algorithm.
                                 For reference, follow this link - https://devtalk.nvidia.com/default/topic/392429/first-cudamalloc-takes-long-time-/ */
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
  matrixNorm_GPU();

  /* Stop Clock */
  gettimeofday(&etstop, &tzdummy);
  etstop2 = times(&cputstop);
  printf("Stopped clock.\n");
  usecstart = (unsigned long long)etstart.tv_sec * 1000000 + etstart.tv_usec;
  usecstop = (unsigned long long)etstop.tv_sec * 1000000 + etstop.tv_usec;

  /* Display output */
  print_B();

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
/* Provided global variables are MAXN, N, A[][] and B[][],
 * defined in the beginning of this code.  B[][] is initialized to zeros.
 */

/* Blank function which is called at first as the first call to any CUDA kernel or function implicitly initializes the CUDA environment.
For reference, follow this link - https://devtalk.nvidia.com/default/topic/392429/first-cudamalloc-takes-long-time-/ */
__global__ void blank_function()
{

}

/* Computes the sums of each group of numbers whose size is denoted by fragSize. */
__global__ void computeSums(float *d_A, float *d_B, size_t pitch_A, size_t pitch_B, int n, int fullBlock, int fragSize, int lastBlockStartRow)
{
    /* Initializing shared memory which will hold the numbers each thread block will be summing. */
    extern __shared__ float chunk[];

    /* Variables to keep track of which number the given is operating on in a 2D perspective. */
    int tx, ty_base, i;
    tx = blockIdx.x;

    /* Calculating the y index of the first element of the numbers being handled by the thread block. */
    if (fullBlock == 0)
    {
        ty_base = threadIdx.x * fragSize + lastBlockStartRow;
    }
    else
    {
        ty_base = (threadIdx.x + blockIdx.y * blockDim.x) * fragSize;
    }

    float *bElem, *aElem;

    /* Getting the starting index of the number the given thread begins operating from on its fragment. This is relative to the chunk[] array. */
    int localStartIndex = threadIdx.x * fragSize;

    /* Copying elements which the block will be summing from global to shared memory of the thread block. */
    for (i = 0; i < fragSize; i++)
    {
        aElem = (float*)((char*)d_A + (pitch_A * (ty_base + i)));
        chunk[localStartIndex + i] = aElem[tx];
    }

    /* Summing up the elements in the block. */
    if (ty_base < (n - 1))
    {
        int inc = 1;
        while(inc < n)
        {
            if ((localStartIndex % (inc * fragSize)) == 0)
            {
                for (i = localStartIndex + inc; (i < localStartIndex + fragSize * inc) && (i < (blockDim.x * fragSize)) && (i < n); i += inc)
                {
                    chunk[localStartIndex] += chunk[i];
                }
                inc *= fragSize;
            }
            else
                break;
            __syncthreads();
        }
    }

    /* Copying result to array d_B on the GPU. */
    for (i = 0; i < fragSize; i++)
    {
        bElem = (float*)((char*)d_B + (pitch_B * (ty_base + i)));
        bElem[tx] = chunk[localStartIndex + i];
    }
}

/* Computes Average Values of the columns in the input 2D array flattened as d_B. */
__global__ void computeFinalSums(float *d_B, size_t pitch_B, int n, int ifLastBlock, int prevBlockSize, int lastBlockStartCol, float *d_Avgs)
{
    int tx, i;
    float *bElem, *bElemBase;

    /* Getting address of starting element of the column which the thread is handling. */
    bElemBase = (float *)((char *)d_B);

    /* Getting x position of the starting element of the column on which the thread begins work. */
    if (ifLastBlock == 0)
    {
        tx = blockIdx.x * blockDim.x + threadIdx.x;
    }
    else
    {
        tx = lastBlockStartCol + threadIdx.x;
    }

    /* Summing up to get final sums for each column and subsequently calculating average. */
    for (i = prevBlockSize; i < n; i += prevBlockSize)
    {
        bElem = (float *)((char *)d_B + (pitch_B * i));
        bElemBase[tx] += bElem[tx];
    }
    bElemBase[tx] /= n;
    d_Avgs[tx] = bElemBase[tx];
}

/* Computes the Variances of each column of input flattened 2D array d_A. */
__global__ void computeVarianceSquares(float *d_A, size_t pitch_A, float *d_Avgs, int n, int ifLastBlock, int lastBlockStartRow)
{
    int tx, ty;
    tx = blockIdx.x;

    /* Getting y position of the starting element of the column on which the thread begins work. */
    if (ifLastBlock == 1)
    {
        ty = threadIdx.x + lastBlockStartRow;
    }
    else
    {
        ty = threadIdx.x + blockIdx.y * blockDim.x;
    }

    /* Getting address of starting element of the column which the thread is handling. */
    float *aElem = (float *)((char *)d_A + (pitch_A * ty));

    /* Calculating and storing the variances in global memory. */
    aElem[tx] = pow((aElem[tx] - d_Avgs[tx]), 2);
}

/* Finally computes the matrix norms for each element in the input flattened 2D matrix d_A. */
__global__ void computeNorms(float *d_A, size_t pitch_A, float *d_Avgs, float *d_Vars, int n, int ifLastBlock, int lastBlockStartRow)
{
    int tx, ty;
    tx = blockIdx.x;

    /* Getting y position of the element on which the thread begins work. */
    if (ifLastBlock == 1)
    {
        ty = threadIdx.x + lastBlockStartRow;
    }
    else
    {
        ty = threadIdx.x + blockIdx.y * blockDim.x;
    }

    float *aElem = (float *)((char *)d_A + (pitch_A * ty));

    /* Calculating and finally storing the norm value of each element. */
    if (d_Vars[tx] != 0)
        aElem[tx] = (aElem[tx] - d_Avgs[tx]) / sqrt(d_Vars[tx]);
    else
        aElem[tx] = 0;
}

/* Self Implemented ceiling function. */
__host__ __device__ int ceil_h_d(float f)
{
    int tmp = (int) f;
	if (f > tmp)
		tmp++;
	return tmp;
}

/* Function to calculate the normalized values of a matrix, implemented on GPU. */
void matrixNorm_GPU()
{
    /* Getting Device Properties of GPU to scale performance according to GPU specs. */
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    int numThreadsPerMP = prop.maxThreadsPerMultiProcessor; /* Getting max. number of concurrent threads each Streaming Multiprocessor in the GPU can run. This is a multiple of the warp size which is 32.*/
    int fragSize = 10; /* The number of elements which are summed by a thread. */
    int BLOCKS_PER_MP = 8; /* Number of threadBlocks each Streaming Multiprocessor supports. */

    int i, j;
    int fullblockSize = numThreadsPerMP / BLOCKS_PER_MP; /* Computing number of threads per thread Block shall run. It should be a multiple of the warp size for maximum efficiency and memory coalescing. */

    int numElemsCol = ceil_h_d((float) N / (float) fragSize); /* Getting number of summing blocks per column */

    int blocksReqdPerCol = ceil_h_d((float) numElemsCol / (float) fullblockSize); /* Getting number of thread blocks required to sum each column. */
    int lastBlockSize = numElemsCol - (blocksReqdPerCol - 1) * fullblockSize; /* Since thread block sizes are multiples of warp size, therefore the last blocksize if not a multiple of warp size, is handled by this thread block. */

    int lastBlockStartRow = (blocksReqdPerCol - 1) * fullblockSize * fragSize; /* Getting the starting row index of the chunk of array being handled by the final thread block */

    float *d_A, *d_Avgs, *d_Vars; /* Declaring device pointers */

    size_t dev_pitch_A;
    size_t host_pitch = N * sizeof(float);

    /* Allocating memory for 2D arrays on GPU */
    cudaMallocPitch(&d_A, &dev_pitch_A, N * sizeof(float), N * sizeof(float));

    /* Allocating memory for 1D arrays on GPU */
    cudaMalloc((void **)&d_Avgs, N * sizeof(float));
    cudaMalloc((void **)&d_Vars, N * sizeof(float));

    /* Allocating memory for flattened 1D array on Host */
    float *A_flat = (float *)malloc(N * N * sizeof(float));

    /* Flattening 2D array on host to a 1D array for transfer to GPU */
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            A_flat[j + i * N] = A[i][j];
        }
    }

    /* Copying flattened 2D array A from host to GPU */
    if (cudaMemcpy2D(d_A, dev_pitch_A, A_flat, host_pitch, N * sizeof(float), N, cudaMemcpyHostToDevice)!= cudaSuccess)
        printf("ERROR");

    dim3 numFullBlocks(N, (blocksReqdPerCol - 1));

    /* Parallel computation of sums of each sum fragment in each column. */
    computeSums<<<numFullBlocks, fullblockSize, (fullblockSize * sizeof(float) * fragSize)>>>(d_A, d_A, dev_pitch_A, dev_pitch_A, N, 1, fragSize, lastBlockStartRow);
    computeSums<<<N, lastBlockSize, (lastBlockSize * sizeof(float) * fragSize)>>>(d_A, d_A, dev_pitch_A, dev_pitch_A, N, 0, fragSize, lastBlockStartRow);


    int numFinalSumBlocksReqd = ceil_h_d((float) N / (float) fullblockSize);
    int lastFinalSumBlockSize = N - (numFinalSumBlocksReqd - 1) * fullblockSize;
    int lastBlockStartCol = (numFinalSumBlocksReqd - 1) * fullblockSize;

    /* Parallel computation of averages of each column which is stored in array d_Avgs. */
    computeFinalSums<<<(numFinalSumBlocksReqd - 1), fullblockSize>>>(d_A, dev_pitch_A, N, 0, (fullblockSize * fragSize), lastBlockStartCol, d_Avgs);
    computeFinalSums<<<1, lastFinalSumBlockSize>>>(d_A, dev_pitch_A, N, 1, (fullblockSize * fragSize), lastBlockStartCol, d_Avgs);

    int numVarBlocksPerCol = ceil_h_d((float) N / (float) fullblockSize);
    int lastVarBlockSize = N - (numVarBlocksPerCol - 1) * fullblockSize;
    int lastVarBlockStartRow = (numVarBlocksPerCol - 1) * fullblockSize;

    dim3 numFullVarBlocks(N, (numVarBlocksPerCol - 1));

    /* Refreshing GPU's copy of array A with original copy from host as it has changed. */
    if (cudaMemcpy2D(d_A, dev_pitch_A, A_flat, host_pitch, N * sizeof(float), N, cudaMemcpyHostToDevice)!= cudaSuccess)
        printf("ERROR");

    /* Parallel computation of variances of each column which is  stored in array d_Vars */
    computeVarianceSquares<<<numFullVarBlocks, fullblockSize>>>(d_A, dev_pitch_A, d_Avgs, N, 0, lastVarBlockStartRow);
    computeVarianceSquares<<<N, lastVarBlockSize>>>(d_A, dev_pitch_A, d_Avgs, N, 1, lastVarBlockStartRow);

    /* Parallel computation of sums of variances in each column. */
    computeSums<<<numFullBlocks, fullblockSize, (fullblockSize * sizeof(float) * fragSize)>>>(d_A, d_A, dev_pitch_A, dev_pitch_A, N, 1, fragSize, lastBlockStartRow);
    computeSums<<<N, lastBlockSize, (lastBlockSize * sizeof(float) * fragSize)>>>(d_A, d_A, dev_pitch_A, dev_pitch_A, N, 0, fragSize, lastBlockStartRow);

    /* Parallel computation of averages of variances in each column */
    computeFinalSums<<<(numFinalSumBlocksReqd - 1), fullblockSize>>>(d_A, dev_pitch_A, N, 0, (fullblockSize * fragSize), lastBlockStartCol, d_Vars);
    computeFinalSums<<<1, lastFinalSumBlockSize>>>(d_A, dev_pitch_A, N, 1, (fullblockSize * fragSize), lastBlockStartCol, d_Vars);

    /* Refreshing GPU's copy of array A with original copy from host as it has changed. */
    if (cudaMemcpy2D(d_A, dev_pitch_A, A_flat, host_pitch, N * sizeof(float), N, cudaMemcpyHostToDevice)!= cudaSuccess)
        printf("ERROR");

    /* Calculating Final Norm values for each column. */
    computeNorms<<<numFullVarBlocks, fullblockSize>>>(d_A, dev_pitch_A, d_Avgs, d_Vars, N, 0, lastVarBlockStartRow);
    computeNorms<<<N, lastVarBlockSize>>>(d_A, dev_pitch_A, d_Avgs, d_Vars, N, 1, lastVarBlockStartRow);

    /* Copying result back to host. */
    cudaMemcpy2D(A_flat, host_pitch, d_A, dev_pitch_A, N * sizeof(float), N, cudaMemcpyDeviceToHost);

    /* Freeing Memory on GPU */
    cudaFree(d_A);
    cudaFree(d_Avgs);
    cudaFree(d_Vars);

    /* Unflattening returned array from 1D to 2D. */
    for (i = 0; i < N; i++) ///Unflattening array returned from GPU
    {
        for (j = 0; j < N; j++)
        {
            B[i][j] = A_flat[j + i * N];
        }
    }
}
