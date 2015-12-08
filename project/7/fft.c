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

#define N_lol 10

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



void parseToComplex(complex target[N][N], float data[N][N])
{
    int a, b;
    for (a = 0; a < N; a++)
    {
        for (b = 0; b < N; b++)
        {
            target[a][b].r = data[a][b];
            target[a][b].i = 0;
        }
    }
}

void parseToReal(float target[N][N], complex data[N][N])
{
    int a, b;
    for (a = 0; a < N; a++)
    {
        for (b = 0; b < N; b++)
        {
            target[a][b] = data[a][b].r;
        }
    }
}

void readFile(const char *fname, float data[N][N])
{
    int i, j;
    FILE *fp;
    fp = fopen(fname, "r");
    for (i = 0; i < N; i++)
    {
        for(j = 0; j < N; j++)
            fscanf(fp, "%g", &data[i][j]);
    }
    fclose(fp);
}

void writeFile(const char *fname, float data[N][N])
{
    int i, j;
    FILE *fp;
    fp = fopen(fname, "w");
    for (i = 0; i < N; i++)
    {
        for(j = 0; j < N; j++)
        {
            fprintf(fp, "%6.2g\t", data[i][j]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

void transpose(complex inp[N_lol][N_lol])
{
    int a, b;
    for (a = 0; a < N_lol; a++)
    {
        for (b = a + 1; b < N_lol; b++)
        {
            C_SWAP(inp[a][b], inp[b][a]);
        }
    }
}

void c_fft2d_serial(complex inp[N][N])
{
    int i, j;
    for (i = 0; i < N; i++)
    {
        c_fft1d(&inp[i][0], N, -1);
    }
    transpose(inp);
    for (i = 0; i < N; i++)
    {
        c_fft1d(&inp[i][0], N, -1);
    }
    transpose(inp);
}

void c_inv_fft2d_serial(complex inp[N][N])
{
    int i, j;
    //transpose(inp);
    for (i = 0; i < N; i++)
    {
        c_fft1d(&inp[i][0], N, +1);
    }

    transpose(inp);
    for (i = 0; i < N; i++)
    {
        c_fft1d(&inp[i][0], N, +1);
    }
    transpose(inp);
}

void pointMul_serial(complex inpA[N][N], complex inpB[N][N])
{
    int a, b;
    for (a = 0; a < N; a++)
    {
        for (b = 0; b < N; b++)
        {
            inpA[a][b].r *= inpB[a][b].r;
            inpA[a][b].i *= inpB[a][b].i;
        }
    }
}

void print2DMatrix(float mat[N_lol][N_lol])
{
    int i, j;
    for (i = 0; i < N_lol; i++)
    {
        for(j = 0; j < N_lol; j++)
        {
            printf("%f\t", mat[i][j]);
        }
        printf("\n");
    }
}

void checkIfEqual(float matA[N][N], float matB[N][N])
{
    int i, j;
    for (i = 0; i < N; i++)
    {
        for(j = 0; j < N; j++)
        {
            if (matA[i][j] != matB[i][j])
            {
                printf("Answer Incorrect :(\nMismatch at i=%d, j=%d\ntarget[%d][%d]=%f, computed[%d][%d]=%f\n", i, j, i, j, matB[i][j], i, j, matA[i][j]);
                return;
            }
        }
    }
    printf("Correct Solution! :D\n");
}


void compute()
{
    float dataA[N][N], dataB[N][N], dataD[N][N], D[N][N];
    //int i, j;

    complex A[N][N], B[N][N];

    readFile("2_im1", dataA);
    readFile("2_im2", dataB);
    readFile("out_2", dataD);

    //print2DMatrix(dataA);

    //readFile("1_im1", dataA);
    //readFile("1_im2", dataB);
    //readFile("out_1", dataD);

    parseToComplex(A, dataA);
    parseToComplex(B, dataB);

    c_fft2d_serial(A);
    c_fft2d_serial(B);

    pointMul_serial(A, B);

    c_inv_fft2d_serial(A);

    parseToReal(D, A);

    //print2DMatrix(D);

    checkIfEqual(D, dataD);
}






void func1(complex *inp)
{
    int i, j;
    for (i = 0; i < N_lol; i++)
        for(j = 0; j < N_lol; j++)
        {
            inp[i * N_lol + j].r = i*N_lol + j;
            inp[i * N_lol + j].i = 0;
        }
}

void func2(complex *inp)
{
    int i, j;
    for (i = 0; i < N_lol; i++)
        for(j = 0; j < N_lol; j++)
        {
            inp[i * N_lol + j].r = (i*N_lol + j)/2;
            inp[i * N_lol + j].i = 0;
        }
}

void func(float inp[N_lol][N_lol])
{
    int i, j;
    for (i = 0; i < N_lol; i++)
        for(j = 0; j < N_lol; j++)
            inp[i][j] = i*N_lol + j;
}

void lolop(complex *arr, int n)
{
    int a, b;
    for(a = 0; a < n; a++)
    {
        /*
        if (arr[a].r > 50)
            arr[a].r = 1;
        else
            arr[a].r = 0;
        */

        arr[a].r *= 2;
        arr[a].i = 3.14;
    }
}

void processFFTChunk(complex *myChunk, int blockSizeRows)
{
    int i, j;
    for (i = 0; i < blockSizeRows; i++)
    {
        lolop(&myChunk[i * N_lol], N_lol);
    }
}


void linearTranspose(complex *data)
{
    int i, j;
    for (i = 0; i < N_lol; i++)
    {
        for (j = i + 1; j < N_lol; j++)
        {
            C_SWAP(data[i * N_lol + j], data[j * N_lol + i]);
        }
    }
}

void c_fft2d_parallel(complex *megaChunk, int nprocs, int blockSizeRows, int lastBlockSize, MPI_Datatype mpi_complex)
{
    int currBlockStart, limit, i, tmp;

    for (tmp = 0; tmp < 2; tmp++)
    {
        for (i = 1; i < nprocs; i++)
        {
            currBlockStart = i * blockSizeRows;
            if ((currBlockStart + blockSizeRows) > N_lol)
                limit = lastBlockSize * N_lol;
            else
                limit = blockSizeRows * N_lol;

            if (currBlockStart < N_lol)
                MPI_Send((megaChunk + currBlockStart * N_lol), limit, mpi_complex, i, 0, MPI_COMM_WORLD);
        }

        for (i = 0; i < blockSizeRows; i++)
        {
            lolop(&megaChunk[i * N_lol], N_lol);
        }

        for (i = 1; i < nprocs - 1; i++)
        {
            MPI_Recv(&megaChunk[i * N_lol * blockSizeRows], blockSizeRows * N_lol, mpi_complex, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        MPI_Recv(&megaChunk[(nprocs - 1) * N_lol * blockSizeRows], lastBlockSize * N_lol, mpi_complex, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        linearTranspose(megaChunk);
    }
}

void pointWiseMul_parallel(complex *megaChunk1, complex *megaChunk2, int nprocs, int blockSizeRows, int lastBlockSize, MPI_Datatype mpi_complex)
{
    int i, j;
    for (i = 1; i < (nprocs - 1); i++)
    {
        MPI_Send((megaChunk1 + i * blockSizeRows * N_lol), blockSizeRows * N_lol, mpi_complex, i, 0, MPI_COMM_WORLD);
        MPI_Send((megaChunk2 + i * blockSizeRows * N_lol), blockSizeRows * N_lol, mpi_complex, i, 1, MPI_COMM_WORLD);
    }
    MPI_Send((megaChunk1 + (nprocs - 1) * blockSizeRows * N_lol), lastBlockSize * N_lol, mpi_complex, i, 0, MPI_COMM_WORLD);
    MPI_Send((megaChunk2 + (nprocs - 1) * blockSizeRows * N_lol), lastBlockSize * N_lol, mpi_complex, i, 1, MPI_COMM_WORLD);

    for (i = 0; i < (blockSizeRows * N_lol); i++)
    {
        megaChunk1[i].r *= megaChunk2[i].r;
        megaChunk1[i].i *= megaChunk2[i].i;
    }

    for (i = 1; i < (nprocs - 1); i++)
    {
        MPI_Recv((megaChunk1 + i * blockSizeRows * N_lol), blockSizeRows * N_lol, mpi_complex, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    MPI_Recv((megaChunk1 + (nprocs - 1) * blockSizeRows * N_lol), lastBlockSize * N_lol, mpi_complex, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}


void processMULChunk(complex *chunk0, complex *chunk1, int blockSizeRows)
{
    int i;
    for (i = 0; i < blockSizeRows * N_lol; i++)
    {
        chunk0[i].r *= chunk1[i].r;
        chunk0[i].i *= chunk1[i].i;
    }
}


void subProcessMethod(int blockSizeRows, MPI_Datatype mpi_complex)
{
    int i;
    complex *myChunk0 = (complex *)malloc(blockSizeRows * N_lol * sizeof(complex));
    complex *myChunk1 = (complex *)malloc(blockSizeRows * N_lol * sizeof(complex));
    for (i = 0; i < 4; i++)
    {
        MPI_Recv(myChunk0, N_lol * blockSizeRows, mpi_complex, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        processFFTChunk(myChunk0, blockSizeRows);
        MPI_Send(myChunk0, blockSizeRows * N_lol, mpi_complex, 0, 0, MPI_COMM_WORLD);
    }
    MPI_Recv(myChunk0, N_lol * blockSizeRows, mpi_complex, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(myChunk1, N_lol * blockSizeRows, mpi_complex, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    processMULChunk(myChunk0, myChunk1, blockSizeRows);
    MPI_Send(myChunk0, blockSizeRows * N_lol, mpi_complex, 0, 0, MPI_COMM_WORLD);
    for (i = 0; i < 2; i++)
    {
        MPI_Recv(myChunk0, N_lol * blockSizeRows, mpi_complex, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        processFFTChunk(myChunk0, blockSizeRows);
        MPI_Send(myChunk0, blockSizeRows * N_lol, mpi_complex, 0, 0, MPI_COMM_WORLD);
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
    blockSizeRows = ceil(N_lol / (float)nprocs);

    int lastBlockSize = blockSizeRows - (blockSizeRows * nprocs - N_lol);

    if (rank == 0)
    {
        printf("Last Block size = %d\n", lastBlockSize);
        complex megaChunk1[N_lol * N_lol], megaChunk2[N_lol * N_lol];
        func1(megaChunk1);
        func2(megaChunk2);
        c_fft2d_parallel(megaChunk2, nprocs, blockSizeRows, lastBlockSize, mpi_complex);
        c_fft2d_parallel(megaChunk1, nprocs, blockSizeRows, lastBlockSize, mpi_complex);

        printf("\n");
        for (i = 0; i < N_lol; i++)
        {
            for (j = 0; j < N_lol; j++)
            {
                printf("%e\t", megaChunk1[i * N_lol + j].r);
            }
            printf("\n");
        }

        printf("\n");
        for (i = 0; i < N_lol; i++)
        {
            for (j = 0; j < N_lol; j++)
            {
                printf("%e\t", megaChunk2[i * N_lol + j].r);
            }
            printf("\n");
        }

        pointWiseMul_parallel(megaChunk1, megaChunk2, nprocs, blockSizeRows, lastBlockSize, mpi_complex);

        printf("\n");
        for (i = 0; i < N_lol; i++)
        {
            for (j = 0; j < N_lol; j++)
            {
                printf("%e\t", megaChunk1[i * N_lol + j].r);
            }
            printf("\n");
        }

        c_fft2d_parallel(megaChunk1, nprocs, blockSizeRows, lastBlockSize, mpi_complex);

        printf("\n");
        for (i = 0; i < N_lol; i++)
        {
            for (j = 0; j < N_lol; j++)
            {
                printf("%e\t", megaChunk1[i * N_lol + j].r);
            }
            printf("\n");
        }
    }

    if (rank > 0 & rank < (nprocs - 1))
    {
        subProcessMethod(blockSizeRows, mpi_complex);
    }

    if (rank == nprocs - 1)
    {
        subProcessMethod(lastBlockSize, mpi_complex);
    }

    MPI_Finalize();
}
