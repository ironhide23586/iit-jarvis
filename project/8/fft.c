/*
 ------------------------------------------------------------------------
 FFT1D            c_fft1d(r,i,-1)
 Inverse FFT1D    c_fft1d(r,i,+1)
 ------------------------------------------------------------------------
*/
/* ---------- FFT 1D
   This computes an in-place complex-to-complex FFT
   r is the real and imaginary arrays of n=2^m points.
   isign = -1 gives forward transform
   isign =  1 gives inverse transform
*/

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

typedef struct {float r; float i;} complex;

static complex ctmp;

#define C_SWAP(a,b) {ctmp=(a);(a)=(b);(b)=ctmp;}
#define N 512

void c_fft1d(complex *r, int      n, int      isign)
{
   int     m,i,i1,j,k,i2,l,l1,l2;
   float   c1,c2,z;
   complex t, u;

   if (isign == 0) return;

   /* Do the bit reversal */
   i2 = n >> 1;
   j = 0;
   for (i=0;i<n-1;i++) {
      if (i < j)
         C_SWAP(r[i], r[j]);
      k = i2;
      while (k <= j) {
         j -= k;
         k >>= 1;
      }
      j += k;
   }

   /* m = (int) log2((double)n); */
   for (i=n,m=0; i>1; m++,i/=2);

   /* Compute the FFT */
   c1 = -1.0;
   c2 =  0.0;
   l2 =  1;
   for (l=0;l<m;l++) {
      l1   = l2;
      l2 <<= 1;
      u.r = 1.0;
      u.i = 0.0;
      for (j=0;j<l1;j++) {
         for (i=j;i<n;i+=l2) {
            i1 = i + l1;

            /* t = u * r[i1] */
            t.r = u.r * r[i1].r - u.i * r[i1].i;
            t.i = u.r * r[i1].i + u.i * r[i1].r;

            /* r[i1] = r[i] - t */
            r[i1].r = r[i].r - t.r;
            r[i1].i = r[i].i - t.i;

            /* r[i] = r[i] + t */
            r[i].r += t.r;
            r[i].i += t.i;
         }
         z =  u.r * c1 - u.i * c2;

         u.i = u.r * c2 + u.i * c1;
         u.r = z;
      }
      c2 = sqrt((1.0 - c1) / 2.0);
      if (isign == -1) /* FWD FFT */
         c2 = -c2;
      c1 = sqrt((1.0 + c1) / 2.0);
   }

   /* Scaling for inverse transform */
   if (isign == 1) {       /* IFFT*/
      for (i=0;i<n;i++) {
         r[i].r /= n;
         r[i].i /= n;
      }
   }
}

void readFileComplex(const char *fname, complex *data)
{
    int i, sz = N * N;
    FILE *fp;
    fp = fopen(fname, "r");
    for (i = 0; i < sz; i++)
    {
        fscanf(fp, "%g", &(data[i].r));
        data[i].i = 0;
    }
    fclose(fp);
}

void writeFileComplex(const char *fname, complex *data)
{
    int i, j;
    FILE *fp;
    fp = fopen(fname, "w");
    for (i = 0; i < N; i++)
    {
        for(j = 0; j < N; j++)
        {
            fprintf(fp, "%6.2g\t", data[i * N + j].r);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

void transpose(complex inp[N][N])
{
    int a, b;
    for (a = 0; a < N; a++)
    {
        for (b = a + 1; b < N; b++)
        {
            C_SWAP(inp[a][b], inp[b][a]);
        }
    }
}

void processFFTChunk(complex *myChunk, int blockSizeRows, int isign)
{
    int i, j;
    for (i = 0; i < blockSizeRows; i++)
    {
        c_fft1d(&myChunk[i * N], N, isign);
    }
}

void linearTranspose(complex *data)
{
    int i, j;
    for (i = 0; i < N; i++)
    {
        for (j = i + 1; j < N; j++)
        {
            C_SWAP(data[i * N + j], data[j * N + i]);
        }
    }
}

void c_fft2d_parallel(complex *megaChunk, int nprocs, int blockSizeRows, int lastBlockSize, MPI_Datatype mpi_complex, int isign)
{
    int currBlockStart, limit, i, tmp;

    for (tmp = 0; tmp < 2; tmp++)
    {
        if (nprocs > 1)
        {
            for (i = 1; i < nprocs; i++)
            {
                currBlockStart = i * blockSizeRows;
                if ((currBlockStart + blockSizeRows) > N)
                    limit = lastBlockSize * N;
                else
                    limit = blockSizeRows * N;

                if (currBlockStart < N)
                    MPI_Send((megaChunk + currBlockStart * N), limit, mpi_complex, i, 0, MPI_COMM_WORLD);
            }
        }


        for (i = 0; i < blockSizeRows; i++)
        {
            c_fft1d(&megaChunk[i * N], N, isign);
        }

        if (nprocs > 1)
        {
            for (i = 1; i < nprocs - 1; i++)
            {
                MPI_Recv(&megaChunk[i * N * blockSizeRows], blockSizeRows * N, mpi_complex, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            MPI_Recv(&megaChunk[(nprocs - 1) * N * blockSizeRows], lastBlockSize * N, mpi_complex, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        linearTranspose(megaChunk);
    }
}

void pointWiseMul_parallel(complex *megaChunk1, complex *megaChunk2, int nprocs, int blockSizeRows, int lastBlockSize, MPI_Datatype mpi_complex)
{
    int i, j;

    if (nprocs > 1)
    {
        for (i = 1; i < (nprocs - 1); i++)
        {
            MPI_Send((megaChunk1 + i * blockSizeRows * N), blockSizeRows * N, mpi_complex, i, 0, MPI_COMM_WORLD);
            MPI_Send((megaChunk2 + i * blockSizeRows * N), blockSizeRows * N, mpi_complex, i, 1, MPI_COMM_WORLD);
        }
        MPI_Send((megaChunk1 + (nprocs - 1) * blockSizeRows * N), lastBlockSize * N, mpi_complex, i, 0, MPI_COMM_WORLD);
        MPI_Send((megaChunk2 + (nprocs - 1) * blockSizeRows * N), lastBlockSize * N, mpi_complex, i, 1, MPI_COMM_WORLD);
    }


    for (i = 0; i < (blockSizeRows * N); i++)
    {
        megaChunk1[i].r *= megaChunk2[i].r;
        megaChunk1[i].i *= megaChunk2[i].i;
    }

    if (nprocs > 1)
    {
        for (i = 1; i < (nprocs - 1); i++)
        {
            MPI_Recv((megaChunk1 + i * blockSizeRows * N), blockSizeRows * N, mpi_complex, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        MPI_Recv((megaChunk1 + (nprocs - 1) * blockSizeRows * N), lastBlockSize * N, mpi_complex, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

void processMULChunk(complex *chunk0, complex *chunk1, int blockSizeRows)
{
    int i;
    for (i = 0; i < blockSizeRows * N; i++)
    {
        chunk0[i].r *= chunk1[i].r;
        chunk0[i].i *= chunk1[i].i;
    }
}

void subProcessMethod(int blockSizeRows, MPI_Datatype mpi_complex)
{
    int i;
    complex *myChunk0 = (complex *)malloc(blockSizeRows * N * sizeof(complex));
    complex *myChunk1 = (complex *)malloc(blockSizeRows * N * sizeof(complex));
    for (i = 0; i < 4; i++)
    {
        MPI_Recv(myChunk0, N * blockSizeRows, mpi_complex, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        processFFTChunk(myChunk0, blockSizeRows, -1);
        MPI_Send(myChunk0, blockSizeRows * N, mpi_complex, 0, 0, MPI_COMM_WORLD);
    }
    MPI_Recv(myChunk0, N * blockSizeRows, mpi_complex, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(myChunk1, N * blockSizeRows, mpi_complex, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    processMULChunk(myChunk0, myChunk1, blockSizeRows);
    MPI_Send(myChunk0, blockSizeRows * N, mpi_complex, 0, 0, MPI_COMM_WORLD);
    for (i = 0; i < 2; i++)
    {
        MPI_Recv(myChunk0, N * blockSizeRows, mpi_complex, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        processFFTChunk(myChunk0, blockSizeRows, +1);
        MPI_Send(myChunk0, blockSizeRows * N, mpi_complex, 0, 0, MPI_COMM_WORLD);
    }
}

int main(int argc, int **argv)
{
    MPI_Init(&argc, &argv);
    int rank, nprocs;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    MPI_Datatype mpi_complex;
    MPI_Type_contiguous(2, MPI_FLOAT, &mpi_complex);
    MPI_Type_commit(&mpi_complex);

    int i, j, blockSizeRows;
    blockSizeRows = ceil(N / (float)nprocs);

    int lastBlockSize = blockSizeRows - (blockSizeRows * nprocs - N);

    if (nprocs == 1)
        lastBlockSize = 0;

    if (rank == 0)
    {
        double startTime, endTime;
        complex megaChunk1[N * N], megaChunk2[N * N];
        char file1[50], file2[50];
        printf("Please enter the input file name. (Please enter after entering a name)-\n");
        scanf("%s", file1);
        scanf("%s", file2);

        printf("Reading Files...\n");
        readFileComplex(file1, megaChunk1);
        readFileComplex(file2, megaChunk2);

        printf("Computing Forward FFT...\n");

        startTime = MPI_Wtime();

        c_fft2d_parallel(megaChunk2, nprocs, blockSizeRows, lastBlockSize, mpi_complex, -1);
        c_fft2d_parallel(megaChunk1, nprocs, blockSizeRows, lastBlockSize, mpi_complex, -1);

        printf("Performing Pointwise Multiplication....\n");
        pointWiseMul_parallel(megaChunk1, megaChunk2, nprocs, blockSizeRows, lastBlockSize, mpi_complex);

        printf("Computing Inverse FFT...\n");
        c_fft2d_parallel(megaChunk1, nprocs, blockSizeRows, lastBlockSize, mpi_complex, +1);

        endTime = MPI_Wtime();

        printf("Writing output to file...\n");
        writeFileComplex("conv_parallel", megaChunk1);

        printf("Execution Time = %f\n", endTime - startTime);
    }

    if (rank > 0 & rank < (nprocs - 1))
        subProcessMethod(blockSizeRows, mpi_complex);

    if (rank == nprocs - 1)
        subProcessMethod(lastBlockSize, mpi_complex);

    MPI_Finalize();
}
