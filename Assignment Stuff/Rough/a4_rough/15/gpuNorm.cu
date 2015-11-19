#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

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
float A[MAXN][MAXN], B[MAXN][MAXN];

float B_GPU[MAXN][MAXN], B_CPU[MAXN][MAXN];

__host__ __device__ int ceil_h_d(float f)
{
    int tmp = (int) f;
	if (f > tmp)
		tmp++;
	return tmp;
}

/* junk */
#define randm() 4|2[uid]&3

/* Prototype */
void matrixNorm_GPU();

/* Prototype */
void matrixNorm();

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
      B_GPU[row][col] = 0.0;
      B_CPU[row][col] = 0.0;
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

    if (N < 10)
    {
        printf("\nB =\n\t");
        for (row = 0; row < N; row++) {
            for (col = 0; col < N; col++) {
                printf("%1.10f%s", B[row][col], (col < N-1) ? ", " : ";\n\t");
            }
        }
    }
}

void print_B_GPU() {
    int row, col;

    if (N < 10)
    {
        printf("\nB_GPU =\n\t");
        for (row = 0; row < N; row++) {
            for (col = 0; col < N; col++) {
                printf("%1.10f%s", B_GPU[row][col], (col < N-1) ? ", " : ";\n\t");
            }
        }
    }
}

void print_B_CPU() {
    int row, col;

    if (N < 10)
    {
        printf("\nB_CPU =\n\t");
        for (row = 0; row < N; row++) {
            for (col = 0; col < N; col++) {
                printf("%1.10f%s", B_CPU[row][col], (col < N-1) ? ", " : ";\n\t");
            }
        }
    }
}

__global__ void computeSums(float *d_A, float *d_B, size_t pitch_A, size_t pitch_B, int n, int fullBlock, int blockSize, int fragSize, int lastBlockStartRow)
{
    extern __shared__ float chunk[];
    int tx, ty_base, i;
    tx = blockIdx.x;
    if (fullBlock == 0)
    {
        ty_base = threadIdx.x * fragSize + lastBlockStartRow;
    }
    else
    {
        ty_base = (threadIdx.x + blockIdx.y * blockSize) * fragSize;
    }

    float *bElem, *aElem;//, *aElemLocalBase, *bElemLocalBase;

    int localStartIndex = threadIdx.x * fragSize;


    for (i = 0; i < fragSize; i++)
    {
        aElem = (float*)((char*)d_A + (pitch_A * (ty_base + i)));
        //aElem = d_A + n *
        chunk[localStartIndex + i] = aElem[tx];
    }

    if (ty_base < (n - 1))
    {
        int inc = 1;
        while(inc < n)
        {
            if ((localStartIndex % (inc * fragSize)) == 0)
            {
                //aElemLocalBase = (float*)((char*)d_A + (pitch_A * (ty_base)));
                //bElemLocalBase = (float*)((char*)d_B + (pitch_B * (ty_base)));
                //bElemLocalBase[tx] = aElemLocalBase[tx];
                for (i = localStartIndex + inc; (i < localStartIndex + fragSize * inc) && (i < (blockSize * fragSize)) && (i < n); i += inc)
                //for (i = localStartIndex + inc; (i < localStartIndex + fragSize * inc) && (i < (blockSize * fragSize)) && (i < n); i += inc)
                {
                    chunk[localStartIndex] += chunk[i];
                    //aElem = (float*)((char*)d_A + (pitch_A * (ty_base + i)));
                    //bElemLocalBase[tx] += aElem[tx];
                }
                inc *= fragSize;
            }
            else
                break;
            __syncthreads();
        }
    }

    for (i = 0; i < fragSize; i++)
    {
        bElem = (float*)((char*)d_B + (pitch_B * (ty_base + i)));
        //aElem = (float*)((char*)d_A + (pitch_A * (ty_base + i)));
        bElem[tx] = chunk[localStartIndex + i];
        //bElem[tx] = aElem[tx];
    }
}

__global__ void computeFinalSums(float *d_B, size_t pitch_B, int n, int ifLastBlock, int prevBlockSize, int lastBlockStartCol, float *d_Avgs)
{
    int tx, i;
    float *bElem, *bElemBase;

    bElemBase = (float *)((char *)d_B);

    if (ifLastBlock == 0)
    {
        tx = blockIdx.x * blockDim.x + threadIdx.x;
    }
    else
    {
        tx = lastBlockStartCol + threadIdx.x;
    }

    for (i = prevBlockSize; i < n; i += prevBlockSize)
    {
        bElem = (float *)((char *)d_B + (pitch_B * i));
        bElemBase[tx] += bElem[tx];
    }
    bElemBase[tx] /= n;
    d_Avgs[tx] = bElemBase[tx];
}

__global__ void computeVarianceSquares(float *d_A, size_t pitch_A, float *d_Avgs, int n, int ifLastBlock, int blockSize, int lastBlockStartRow)
{
    int tx, ty;
    tx = blockIdx.x;
    if (ifLastBlock == 1)
    {
        ty = threadIdx.x + lastBlockStartRow;
    }
    else
    {
        ty = threadIdx.x + blockIdx.y * blockSize;
    }

    float *aElem = (float *)((char *)d_A + (pitch_A * ty));

    aElem[tx] = pow((aElem[tx] - d_Avgs[tx]), 2);
}

__global__ void computeNorms(float *d_A, size_t pitch_A, float *d_Avgs, float *d_Vars, int n, int ifLastBlock, int blockSize, int lastBlockStartRow)
{
    int tx, ty;
    tx = blockIdx.x;
    if (ifLastBlock == 1)
    {
        ty = threadIdx.x + lastBlockStartRow;
    }
    else
    {
        ty = threadIdx.x + blockIdx.y * blockSize;
    }

    float *aElem = (float *)((char *)d_A + (pitch_A * ty));

    if (d_Vars[tx] != 0)
        aElem[tx] = (aElem[tx] - d_Avgs[tx])/sqrt(d_Vars[tx]);
    else
        aElem[tx] = 0;
}

__global__ void dummy()
{

}

int main(int argc, char **argv)
{
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

    dummy<<<1, 1>>>(); ///DEVICE INITIALIZATION

    /* Start GPU Clock */
    printf("\nStarting GPU clock.\n");
    gettimeofday(&etstart, &tzdummy);
    etstart2 = times(&cputstart);

    matrixNorm_GPU();

    /* Stop GPU Clock */
    gettimeofday(&etstop, &tzdummy);
    etstop2 = times(&cputstop);
    printf("Stopped GPU clock.\n");
    usecstart = (unsigned long long)etstart.tv_sec * 1000000 + etstart.tv_usec;
    usecstop = (unsigned long long)etstop.tv_sec * 1000000 + etstop.tv_usec;

    /* Display output */
    print_B_GPU();

    /* Display timing results */
    float t_GPU = (float)(usecstop - usecstart)/(float)1000;
    printf("\nElapsed GPU time = %g ms.\n",
    (float)(usecstop - usecstart)/(float)1000);

    printf("\n===========================================================================\n");

    /* Print input matrices */
    print_inputs();

    /* Start CPU Clock */
    printf("\nStarting CPU clock.\n");
    gettimeofday(&etstart, &tzdummy);
    etstart2 = times(&cputstart);

    matrixNorm();

    /* Stop CPU Clock */
    gettimeofday(&etstop, &tzdummy);
    etstop2 = times(&cputstop);
    printf("Stopped CPU clock.\n");
    usecstart = (unsigned long long)etstart.tv_sec * 1000000 + etstart.tv_usec;
    usecstop = (unsigned long long)etstop.tv_sec * 1000000 + etstop.tv_usec;

    /* Display output */
    print_B_CPU();

    /* Display timing results */
    printf("\nElapsed CPU time = %g ms.\n",
    (float)(usecstop - usecstart)/(float)1000);

    float t_CPU = (float)(usecstop - usecstart)/(float)1000;

    /* Error Checking */
    int k = 0, i, j;
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            if (abs(B_GPU[i][j]-B_CPU[i][j]) > 0.000005f)
            {
                printf("\nMatrices Unequal. Unequality at row %d, col %d;\nB_GPU[%d][%d]=%f, B_CPU[%d][%d]=%f\n", i, j, i, j, B_GPU[i][j], i, j, B_CPU[i][j]);
                i = N;
                j = N;
                k = 1;
                break;
            }

        }
    }
    if (k == 0)
        printf("\nArray B_GPU & B_CPU are equal!!! :D\n");

    float speedup = t_CPU / t_GPU;
    printf("\nSpeedup = %f\n", speedup);
}

void matrixNorm_GPU()
{
    ///SEGMENTATION FAULT AT N=1254

    int fragSize = 10;
    int BLOCKS_PER_MP = 8;
    //int numThreadsPerMP = 1536;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    int numThreadsPerMP = prop.maxThreadsPerMultiProcessor;

    int fullblockSize = numThreadsPerMP / BLOCKS_PER_MP;

    int i, j;

    /* Timing variables */
    struct timeval etstart, etstop;  /* Elapsed times using gettimeofday() */
    struct timezone tzdummy;
    clock_t etstart2, etstop2;  /* Elapsed times using times() */
    unsigned long long usecstart, usecstop;
    struct tms cputstart, cputstop;  /* CPU times for my processes */

    int numElemsCol = ceil_h_d((float) N / (float) fragSize);

    int blocksReqdPerCol = ceil_h_d((float) numElemsCol / (float) fullblockSize);
    int lastBlockSize = numElemsCol - (blocksReqdPerCol - 1) * fullblockSize;

    int lastBlockStartRow = (blocksReqdPerCol - 1) * fullblockSize * fragSize;

    float *d_A, *d_B, *d_Avgs, *d_Vars;

    size_t dev_pitch_A, dev_pitch_B;
    size_t host_pitch = N * sizeof(float);

    /* Start Clock */
    //printf("\nStarting GPU clock.\n");
    gettimeofday(&etstart, &tzdummy);
    etstart2 = times(&cputstart);

    cudaMallocPitch(&d_A, &dev_pitch_A, N * sizeof(float), N * sizeof(float));
    cudaMallocPitch(&d_B, &dev_pitch_B, N * sizeof(float), N * sizeof(float));

    //cudaMalloc((void **)&d_A, (N * N * sizeof(float)));
    //cudaMalloc((void **)&d_B, (N * N * sizeof(float)));

    /* Stop Clock */
    gettimeofday(&etstop, &tzdummy);
    etstop2 = times(&cputstop);
    //printf("Stopped GPU clock.\n");
    usecstart = (unsigned long long)etstart.tv_sec * 1000000 + etstart.tv_usec;
    usecstop = (unsigned long long)etstop.tv_sec * 1000000 + etstop.tv_usec;
    /* Display timing results */
    printf("\nElapsed GPU 2D Memory Allocation time = %g ms.\n",
    (float)(usecstop - usecstart)/(float)1000);







    /* Start Clock */
    //printf("\nStarting GPU clock.\n");
    gettimeofday(&etstart, &tzdummy);
    etstart2 = times(&cputstart);

    cudaMalloc((void **)&d_Avgs, N * sizeof(float));
    cudaMalloc((void **)&d_Vars, N * sizeof(float));

    /* Stop Clock */
    gettimeofday(&etstop, &tzdummy);
    etstop2 = times(&cputstop);
    //printf("Stopped GPU clock.\n");
    usecstart = (unsigned long long)etstart.tv_sec * 1000000 + etstart.tv_usec;
    usecstop = (unsigned long long)etstop.tv_sec * 1000000 + etstop.tv_usec;
    /* Display timing results */
    printf("\nElapsed GPU 1D Memory Allocation time = %g ms.\n",
    (float)(usecstop - usecstart)/(float)1000);






    float A_flat[N * N], B_flat[N * N];



    /* Start Clock */
    //printf("\nStarting GPU clock.\n");
    gettimeofday(&etstart, &tzdummy);
    etstart2 = times(&cputstart);

    for (i = 0; i < N; i++) ///Flattening out Array for transfer to GPU
    {
        for (j = 0; j < N; j++)
        {
            A_flat[j + i * N] = A[i][j];
        }
    }

    /* Stop Clock */
    gettimeofday(&etstop, &tzdummy);
    etstop2 = times(&cputstop);
    //printf("Stopped GPU clock.\n");
    usecstart = (unsigned long long)etstart.tv_sec * 1000000 + etstart.tv_usec;
    usecstop = (unsigned long long)etstop.tv_sec * 1000000 + etstop.tv_usec;
    /* Display timing results */
    printf("\nElapsed A flattening time = %g ms.\n",
    (float)(usecstop - usecstart)/(float)1000);






    /* Start Clock */
    //printf("\nStarting GPU clock.\n");
    gettimeofday(&etstart, &tzdummy);
    etstart2 = times(&cputstart);

    if (cudaMemcpy2D(d_A, dev_pitch_A, A_flat, host_pitch, N * sizeof(float), N, cudaMemcpyHostToDevice)!= cudaSuccess)
        printf("ERROR");

    /* Stop Clock */
    gettimeofday(&etstop, &tzdummy);
    etstop2 = times(&cputstop);
    //printf("Stopped GPU clock.\n");
    usecstart = (unsigned long long)etstart.tv_sec * 1000000 + etstart.tv_usec;
    usecstop = (unsigned long long)etstop.tv_sec * 1000000 + etstop.tv_usec;
    /* Display timing results */
    printf("\nElapsed A transfer to GPU time = %g ms.\n",
    (float)(usecstop - usecstart)/(float)1000);





    dim3 numFullBlocks(N, (blocksReqdPerCol - 1)); ///N cols, fullBlockReqd rows





    /* Start Clock */
    //printf("\nStarting GPU clock.\n");
    gettimeofday(&etstart, &tzdummy);
    etstart2 = times(&cputstart);

    computeSums<<<numFullBlocks, fullblockSize, (fullblockSize * sizeof(float) * fragSize)>>>(d_A, d_B, dev_pitch_A, dev_pitch_B, N, 1, fullblockSize, fragSize, lastBlockStartRow);
    computeSums<<<N, lastBlockSize, (lastBlockSize * sizeof(float) * fragSize)>>>(d_A, d_B, dev_pitch_A, dev_pitch_B, N, 0, lastBlockSize, fragSize, lastBlockStartRow);

    //computeSums<<<numFullBlocks, fullblockSize>>>(d_A, d_B, dev_pitch_A, dev_pitch_B, N, 1, fullblockSize, fragSize, lastBlockStartRow);
    //computeSums<<<N, lastBlockSize>>>(d_A, d_B, dev_pitch_A, dev_pitch_B, N, 0, lastBlockSize, fragSize, lastBlockStartRow);

    /* Stop Clock */
    gettimeofday(&etstop, &tzdummy);
    etstop2 = times(&cputstop);
    //printf("Stopped GPU clock.\n");
    usecstart = (unsigned long long)etstart.tv_sec * 1000000 + etstart.tv_usec;
    usecstop = (unsigned long long)etstop.tv_sec * 1000000 + etstop.tv_usec;
    /* Display timing results */
    printf("\nElapsed computeSums time = %g ms.\n",
    (float)(usecstop - usecstart)/(float)1000);




    int numFinalSumBlocksReqd = ceil_h_d((float) N / (float) fullblockSize);
    int lastFinalSumBlockSize = N - (numFinalSumBlocksReqd - 1) * fullblockSize;
    int lastBlockStartCol = (numFinalSumBlocksReqd - 1) * fullblockSize;




    /* Start Clock */
    //printf("\nStarting GPU clock.\n");
    gettimeofday(&etstart, &tzdummy);
    etstart2 = times(&cputstart);

    computeFinalSums<<<(numFinalSumBlocksReqd - 1), fullblockSize>>>(d_B, dev_pitch_B, N, 0, (fullblockSize * fragSize), lastBlockStartCol, d_Avgs);
    computeFinalSums<<<1, lastFinalSumBlockSize>>>(d_B, dev_pitch_B, N, 1, (fullblockSize * fragSize), lastBlockStartCol, d_Avgs);

    /* Stop Clock */
    gettimeofday(&etstop, &tzdummy);
    etstop2 = times(&cputstop);
    //printf("Stopped GPU clock.\n");
    usecstart = (unsigned long long)etstart.tv_sec * 1000000 + etstart.tv_usec;
    usecstop = (unsigned long long)etstop.tv_sec * 1000000 + etstop.tv_usec;
    /* Display timing results */
    printf("\nElapsed computeFinalSums time = %g ms.\n",
    (float)(usecstop - usecstart)/(float)1000);






    int numVarBlocksPerCol = ceil_h_d((float) N / (float) fullblockSize);
    int lastVarBlockSize = N - (numVarBlocksPerCol - 1) * fullblockSize;
    int lastVarBlockStartRow = (numVarBlocksPerCol - 1) * fullblockSize;

    dim3 numFullVarBlocks(N, (numVarBlocksPerCol - 1));





    /* Start Clock */
    //printf("\nStarting GPU clock.\n");
    gettimeofday(&etstart, &tzdummy);
    etstart2 = times(&cputstart);

    computeVarianceSquares<<<numFullVarBlocks, fullblockSize>>>(d_A, dev_pitch_A, d_Avgs, N, 0, fullblockSize, lastVarBlockStartRow);
    computeVarianceSquares<<<N, lastVarBlockSize>>>(d_A, dev_pitch_A, d_Avgs, N, 1, lastVarBlockSize, lastVarBlockStartRow);

    /* Stop Clock */
    gettimeofday(&etstop, &tzdummy);
    etstop2 = times(&cputstop);
    //printf("Stopped GPU clock.\n");
    usecstart = (unsigned long long)etstart.tv_sec * 1000000 + etstart.tv_usec;
    usecstop = (unsigned long long)etstop.tv_sec * 1000000 + etstop.tv_usec;
    /* Display timing results */
    printf("\nElapsed computeVarianceSquares time = %g ms.\n",
    (float)(usecstop - usecstart)/(float)1000);




    /* Start Clock */
    //printf("\nStarting GPU clock.\n");
    gettimeofday(&etstart, &tzdummy);
    etstart2 = times(&cputstart);

    computeSums<<<numFullBlocks, fullblockSize, (fullblockSize * sizeof(float) * fragSize)>>>(d_A, d_B, dev_pitch_A, dev_pitch_B, N, 1, fullblockSize, fragSize, lastBlockStartRow);
    computeSums<<<N, lastBlockSize, (lastBlockSize * sizeof(float) * fragSize)>>>(d_A, d_B, dev_pitch_A, dev_pitch_B, N, 0, lastBlockSize, fragSize, lastBlockStartRow);

    //computeSums<<<numFullBlocks, fullblockSize>>>(d_A, d_B, dev_pitch_A, dev_pitch_B, N, 1, fullblockSize, fragSize, lastBlockStartRow);
    //computeSums<<<N, lastBlockSize>>>(d_A, d_B, dev_pitch_A, dev_pitch_B, N, 0, lastBlockSize, fragSize, lastBlockStartRow);

    /* Stop Clock */
    gettimeofday(&etstop, &tzdummy);
    etstop2 = times(&cputstop);
    //printf("Stopped GPU clock.\n");
    usecstart = (unsigned long long)etstart.tv_sec * 1000000 + etstart.tv_usec;
    usecstop = (unsigned long long)etstop.tv_sec * 1000000 + etstop.tv_usec;
    /* Display timing results */
    printf("\nElapsed computeSums (Variance Summing) time = %g ms.\n",
    (float)(usecstop - usecstart)/(float)1000);





    /* Start Clock */
    //printf("\nStarting GPU clock.\n");
    gettimeofday(&etstart, &tzdummy);
    etstart2 = times(&cputstart);

    computeFinalSums<<<(numFinalSumBlocksReqd - 1), fullblockSize>>>(d_B, dev_pitch_B, N, 0, (fullblockSize * fragSize), lastBlockStartCol, d_Vars);
    computeFinalSums<<<1, lastFinalSumBlockSize>>>(d_B, dev_pitch_B, N, 1, (fullblockSize * fragSize), lastBlockStartCol, d_Vars);

    /* Stop Clock */
    gettimeofday(&etstop, &tzdummy);
    etstop2 = times(&cputstop);
    //printf("Stopped GPU clock.\n");
    usecstart = (unsigned long long)etstart.tv_sec * 1000000 + etstart.tv_usec;
    usecstop = (unsigned long long)etstop.tv_sec * 1000000 + etstop.tv_usec;
    /* Display timing results */
    printf("\nElapsed computeFinalSums (Variance Final Average) time = %g ms.\n",
    (float)(usecstop - usecstart)/(float)1000);






    /* Start Clock */
    //printf("\nStarting GPU clock.\n");
    gettimeofday(&etstart, &tzdummy);
    etstart2 = times(&cputstart);

    if (cudaMemcpy2D(d_A, dev_pitch_A, A_flat, host_pitch, N * sizeof(float), N, cudaMemcpyHostToDevice)!= cudaSuccess)
        printf("ERROR");

    /* Stop Clock */
    gettimeofday(&etstop, &tzdummy);
    etstop2 = times(&cputstop);
    //printf("Stopped GPU clock.\n");
    usecstart = (unsigned long long)etstart.tv_sec * 1000000 + etstart.tv_usec;
    usecstop = (unsigned long long)etstop.tv_sec * 1000000 + etstop.tv_usec;
    /* Display timing results */
    printf("\nElapsed A GPU refresh after variance calculation time = %g ms.\n",
    (float)(usecstop - usecstart)/(float)1000);








    /* Start Clock */
    //printf("\nStarting GPU clock.\n");
    gettimeofday(&etstart, &tzdummy);
    etstart2 = times(&cputstart);

    computeNorms<<<numFullVarBlocks, fullblockSize>>>(d_A, dev_pitch_A, d_Avgs, d_Vars, N, 0, fullblockSize, lastVarBlockStartRow);
    computeNorms<<<N, lastVarBlockSize>>>(d_A, dev_pitch_A, d_Avgs, d_Vars, N, 1, lastVarBlockSize, lastVarBlockStartRow);

    /* Stop Clock */
    gettimeofday(&etstop, &tzdummy);
    etstop2 = times(&cputstop);
    //printf("Stopped GPU clock.\n");
    usecstart = (unsigned long long)etstart.tv_sec * 1000000 + etstart.tv_usec;
    usecstop = (unsigned long long)etstop.tv_sec * 1000000 + etstop.tv_usec;
    /* Display timing results */
    printf("\nElapsed computeNorms time = %g ms.\n",
    (float)(usecstop - usecstart)/(float)1000);




    /* Start Clock */
    //printf("\nStarting GPU clock.\n");
    gettimeofday(&etstart, &tzdummy);
    etstart2 = times(&cputstart);

    cudaMemcpy2D(B_flat, host_pitch, d_A, dev_pitch_A, N * sizeof(float), N, cudaMemcpyDeviceToHost);

    /* Stop Clock */
    gettimeofday(&etstop, &tzdummy);
    etstop2 = times(&cputstop);
    //printf("Stopped GPU clock.\n");
    usecstart = (unsigned long long)etstart.tv_sec * 1000000 + etstart.tv_usec;
    usecstop = (unsigned long long)etstop.tv_sec * 1000000 + etstop.tv_usec;
    /* Display timing results */
    printf("\nElapsed B transfer to CPU time = %g ms.\n",
    (float)(usecstop - usecstart)/(float)1000);





    /* Start Clock */
    //printf("\nStarting GPU clock.\n");
    gettimeofday(&etstart, &tzdummy);
    etstart2 = times(&cputstart);

    for (i = 0; i < N; i++) ///Unflattening array returned from GPU
    {
        for (j = 0; j < N; j++)
        {
            B_GPU[i][j] = B_flat[j + i * N];
        }
    }

    /* Stop Clock */
    gettimeofday(&etstop, &tzdummy);
    etstop2 = times(&cputstop);
    //printf("Stopped GPU clock.\n");
    usecstart = (unsigned long long)etstart.tv_sec * 1000000 + etstart.tv_usec;
    usecstop = (unsigned long long)etstop.tv_sec * 1000000 + etstop.tv_usec;
    /* Display timing results */
    printf("\nElapsed B unflattening time = %g ms.\n",
    (float)(usecstop - usecstart)/(float)1000);
}


void matrixNorm() {
  int row, col;
  float mu, sigma; // Mean and Standard Deviation

  printf("\nComputing Serially.\n");

    for (col=0; col < N; col++) {
        mu = 0.0;
        for (row=0; row < N; row++)
            mu += A[row][col];
        mu /= (float) N;
        sigma = 0.0;
        for (row=0; row < N; row++)
            sigma += powf(A[row][col] - mu, 2.0);
        sigma /= (float) N;
        sigma = sqrt(sigma);
        for (row=0; row < N; row++) {
            if (sigma == 0.0)
                B_CPU[row][col] = 0.0;
            else
                B_CPU[row][col] = (A[row][col] - mu) / sigma;
        }
    }

}
