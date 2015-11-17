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

#define BLOCKS_PER_MP 8

/* Program Parameters */
#define MAXN 8000  /* Max value of N */
int N;  /* Matrix size */
float A[MAXN][MAXN], B[MAXN][MAXN];

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


__global__ void testKernel(float *d_A, float *d_B, size_t pitch_A, size_t pitch_B, int n, int fullBlock, int blockSize, int fragSize)
{
    //if (fullBlock == 0)
    {
        int tx = blockIdx.x;
        int ty = threadIdx.x * fragSize;

        //float* bElem = (float*)((char*)d_B + (pitch_B * tx));
        float* aElem = (float*)(((char*)d_A) + (n*ty));
        //if (ty <= 1)

        printf("pitch = %d, start = %f\n", pitch_A, *(aElem));
        printf("Element at (%d, %d) = %f\n", tx, ty, aElem[tx]);

        //bElem[tx] = aElem[tx];
    }

    //int idx = blockIdx.x * blockDim.x + threadIdx.x;

    //int row = idx/n;

    //int col = idx%n;

    //d_B[row][col] = d_A[row][col];
}

int main(int argc, char **argv)
{
    /* Process program parameters */
    parameters(argc, argv);

    /* Initialize A and B */
    initialize_inputs();

    /* Print input matrices */
    print_inputs();

    matrixNorm_GPU();
    //matrixNorm();

    /* Display output */
    print_B();
}

void matrixNorm_GPU()
{
    //cudaDeviceProp prop;
    //cudaGetDeviceProperties(&prop, 0);

    //int numMP = prop.multiProcessorCount;
    //int numThreadsPerMP = prop.maxThreadsPerMultiProcessor;
    //int warpSize = prop.warpSize;

    int numMP = 15;
    int numThreadsPerMP = 1536;
    int warpSize = 32;
    int fragSize = 2;

    int blockSize = numThreadsPerMP / BLOCKS_PER_MP;

    int numElemsCol = ceil_h_d((float) N / (float) fragSize);

    printf("CEIL(N/%d) = %d\n\n", fragSize, numElemsCol);

    int blocksReqdPerCol = ceil_h_d((float) numElemsCol / (float) blockSize);
    int lastBlockSize = numElemsCol - (blocksReqdPerCol - 1) * blockSize;

    float *d_A, *d_B;

    size_t dev_pitch_A, dev_pitch_B;
    size_t host_pitch = N * sizeof(float);

    cudaMallocPitch(&d_A, &dev_pitch_A, N * sizeof(float), N * sizeof(float));
    cudaMallocPitch(&d_B, &dev_pitch_B, N * sizeof(float), N * sizeof(float));

    cudaMemcpy2D(d_A, dev_pitch_A, A, host_pitch, N * sizeof(float), N, cudaMemcpyHostToDevice);

    dim3 numFullBlocks(N, (blocksReqdPerCol - 1)); ///N cols, fullBlockReqd rows

    testKernel<<<numFullBlocks, blockSize>>>(d_A, d_B, dev_pitch_A, dev_pitch_B, N, 1, blockSize, fragSize);
    testKernel<<<N, lastBlockSize>>>(d_A, d_B, dev_pitch_A, dev_pitch_B, N, 0, lastBlockSize, fragSize);

    cudaMemcpy2D(B, host_pitch, d_B, dev_pitch_B, N * sizeof(float), N, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    //printf("No. of MPs - %d\n", numMP);
    //printf("No. of threads per MP - %d\n", numThreadsPerMP);
    //printf("Warp Size - %d\n", warpSize);
    //printf("Block Size - %d\n\n", blockSize);

    printf("Blocks Reqd - %d\n", blocksReqdPerCol);
    printf("Last Block Size - %d\n", lastBlockSize);
}


void matrixNorm() {
  int row, col;
  float mu, sigma; // Mean and Standard Deviation

  printf("Computing Serially.\n");

    for (col=0; col < N; col++) {
        mu = 0.0;
        for (row=0; row < N; row++)
            mu += A[row][col];
        mu /= (float) N;
        sigma = 0.0;
        for (row=0; row < N; row++)
            sigma += powf(A[row][col] - mu, 2.0);
        sigma /= (float) N;
        for (row=0; row < N; row++) {
            if (sigma == 0.0)
                B[row][col] = 0.0;
            else
                B[row][col] = (A[row][col] - mu) / sigma;
        }
    }

}
