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
#define GROUP_SIZE 2

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

void processFFTChunk(complex *myChunk, int blockSizeRows, int isign)
{
    int i;
    for (i = 0; i < blockSizeRows; i++)
    {
        c_fft1d(&myChunk[i * N], N, isign);
    }
}

void c_fft2d_parallel(complex *megaChunk0, complex *myChunk0, int blockSizeRows, int absoluteBlockSize, MPI_Datatype mpi_complex, int isign, double *commDuration, int rank, MPI_Comm myComm, int rootRank)
{
    int i, j, k;
    double startTime, endTime;
    *commDuration = 0;
    for (k = 0; k < 2; k++)
    {
        startTime = MPI_Wtime();
        MPI_Scatter(megaChunk0, absoluteBlockSize, mpi_complex, myChunk0, absoluteBlockSize, mpi_complex, 0, myComm);
        endTime = MPI_Wtime();
        *commDuration += (endTime - startTime);

        if (MPI_COMM_NULL != myComm)
            processFFTChunk(myChunk0, blockSizeRows, isign);

        startTime = MPI_Wtime();
        MPI_Gather(myChunk0, absoluteBlockSize, mpi_complex, megaChunk0, absoluteBlockSize, mpi_complex, 0, myComm);
        endTime = MPI_Wtime();
        *commDuration += (endTime - startTime);

        if (rank == rootRank)
            linearTranspose(megaChunk0);
    }
}

void processMULChunk(complex *chunk0, complex *chunk1, int absoluteBlockSize)
{
    int i;
    for (i = 0; i < absoluteBlockSize; i++)
    {
        chunk0[i].r *= chunk1[i].r;
        chunk0[i].i *= chunk1[i].i;
    }
}

void pointWiseMul_parallel(complex *megaChunk0, complex *megaChunk1, complex *myChunk0, complex *myChunk1, int absoluteBlockSize, MPI_Datatype mpi_complex, double *commDuration, MPI_Comm myComm)
{
    int i, j;
    double startTime, endTime;
    *commDuration = 0;

    startTime = MPI_Wtime();
    MPI_Scatter(megaChunk0, absoluteBlockSize, mpi_complex, myChunk0, absoluteBlockSize, mpi_complex, 0, myComm);
    MPI_Scatter(megaChunk1, absoluteBlockSize, mpi_complex, myChunk1, absoluteBlockSize, mpi_complex, 0, myComm);
    endTime = MPI_Wtime();
    *commDuration += (endTime - startTime);

    processMULChunk(myChunk0, myChunk1, absoluteBlockSize);

    startTime = MPI_Wtime();
    MPI_Gather(myChunk0, absoluteBlockSize, mpi_complex, megaChunk0, absoluteBlockSize, mpi_complex, 0, myComm);
    endTime = MPI_Wtime();
    *commDuration += (endTime - startTime);
}


int main(int argc, int **argv)
{
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    int i, j, blockSizeRows, absoluteBlockSize, totalSize = N * N;
    double startTime, endTime, comm_Duration = 0, tmp_duration, totalDuration, computeDuration, startTimeGlobal, endTimeGlobal;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    MPI_Datatype mpi_complex;
    MPI_Type_contiguous(2, MPI_FLOAT, &mpi_complex);
    MPI_Type_commit(&mpi_complex);

    int color = rank / GROUP_SIZE;
    MPI_Comm myComm;
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &myComm);
    int myRank;
    MPI_Comm_rank(myComm, &myRank);

    blockSizeRows = N / GROUP_SIZE;
    absoluteBlockSize = blockSizeRows * N;

    complex *myChunk0 = (complex *)malloc(absoluteBlockSize * sizeof(complex));
    complex *myChunk1 = (complex *)malloc(absoluteBlockSize * sizeof(complex));

    complex megaChunk0[N * N], megaChunk1[N * N];

    if (rank == 0)
    {
        char file0[50], file1[50];

        printf("Please enter the input file names. (Press enter after entering a name)-\n");
        scanf("%s", file0);
        scanf("%s", file1);

        printf("Reading Files...\n");
        readFileComplex(file0, megaChunk0);
        readFileComplex(file1, megaChunk1);

        printf("Computing Forward FFT...\n");

        startTime = MPI_Wtime();
        startTimeGlobal = startTime;
        MPI_Send(megaChunk1, totalSize, mpi_complex, 2, 0, MPI_COMM_WORLD);
        endTime = MPI_Wtime();
        comm_Duration += (endTime - startTime);
    }

    if (rank < 2)
    {
        c_fft2d_parallel(megaChunk0, myChunk0, blockSizeRows, absoluteBlockSize, mpi_complex, -1, &tmp_duration, rank, myComm, 0);

        if (rank == 0)
        {
            startTime = MPI_Wtime();
            MPI_Send(megaChunk0, totalSize, mpi_complex, 4, 0, MPI_COMM_WORLD);
            endTime = MPI_Wtime();
            comm_Duration += (endTime - startTime);

            MPI_Recv(&tmp_duration, 1, MPI_DOUBLE, 2, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            comm_Duration += tmp_duration;
            MPI_Recv(&tmp_duration, 1, MPI_DOUBLE, 4, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            comm_Duration += tmp_duration;
            MPI_Recv(&tmp_duration, 1, MPI_DOUBLE, 6, 6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            comm_Duration += tmp_duration;

            endTimeGlobal = MPI_Wtime();
            totalDuration = endTimeGlobal - startTimeGlobal;
            computeDuration = totalDuration - comm_Duration;

            printf("Total Time Taken = %f\n", totalDuration);
            printf("Communication Time = %f\n", comm_Duration);
            printf("Computation Time = %f\n", computeDuration);
        }
    }


    if (rank > 1 && rank < 4)
    {
        if (rank == 2)
            MPI_Recv(megaChunk0, totalSize, mpi_complex, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        c_fft2d_parallel(megaChunk0, myChunk0, blockSizeRows, absoluteBlockSize, mpi_complex, -1, &tmp_duration, rank, myComm, 2);

        if (rank == 2)
        {
            startTime = MPI_Wtime();
            MPI_Send(megaChunk0, totalSize, mpi_complex, 4, 1, MPI_COMM_WORLD);
            endTime = MPI_Wtime();
            tmp_duration += (endTime - startTime);

            MPI_Send(&tmp_duration, 1, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
        }
    }


    if (rank > 3 && rank < 6)
    {
        if (rank == 4)
        {
            MPI_Recv(megaChunk0, totalSize, mpi_complex, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(megaChunk1, totalSize, mpi_complex, 2, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("Performing point-wise multiplication...\n");
        }

        pointWiseMul_parallel(megaChunk0, megaChunk1, myChunk0, myChunk1, absoluteBlockSize, mpi_complex, &tmp_duration, myComm);

        if (rank == 4)
        {
            startTime = MPI_Wtime();
            MPI_Send(megaChunk0, totalSize, mpi_complex, 6, 0, MPI_COMM_WORLD);
            endTime = MPI_Wtime();
            tmp_duration += (endTime - startTime);

            MPI_Send(&tmp_duration, 1, MPI_DOUBLE, 0, 4, MPI_COMM_WORLD);
        }
    }

    if (rank > 5 && rank < 8)
    {
        if (rank == 6)
        {
            MPI_Recv(megaChunk0, totalSize, mpi_complex, 4, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("Computing Inverse FFT...\n");
        }

        MPI_Barrier(myComm);
        c_fft2d_parallel(megaChunk0, myChunk0, blockSizeRows, absoluteBlockSize, mpi_complex, +1, &tmp_duration, rank, myComm, 6);

        if (rank == 6)
        {
            MPI_Send(&tmp_duration, 1, MPI_DOUBLE, 0, 6, MPI_COMM_WORLD);

            printf("Writing output files...\n");
            writeFileComplex("conv_parallel_d", megaChunk0);
        }
    }

    MPI_Comm_free(&myComm);
    MPI_Finalize();
}
