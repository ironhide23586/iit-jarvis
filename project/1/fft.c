
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
//#include <mpi.h>

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

void transpose(complex inp[N][N])
{
    int a, b;
    float tmp_r, tmp_i;
    for (a = 0; a < N; a++)
    {
        for (b = a + 1; b < N; b++)
        {
            //C_SWAP(inp[a][b], inp[b][a]);
            tmp_r = inp[a][b].r;
            inp[a][b].r = inp[b][a].r;
            inp[b][a].r = tmp_r;

            tmp_i = inp[a][b].i;
            inp[a][b].i = inp[b][a].i;
            inp[b][a].i = tmp_i;
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

void print2DMatrix(float mat[N][N])
{
    int i, j;
    for (i = 0; i < N; i++)
    {
        for(j = 0; j < N; j++)
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

int main()
{
    float dataA[N][N], dataB[N][N], dataD[N][N];
    //int i, j;

    complex A[N][N], B[N][N];

    readFile("1_im1", dataA);
    readFile("1_im2", dataB);
    readFile("out_1", dataD);

    //print2DMatrix(dataA);

    //readFile("1_im1", dataA);
    //readFile("1_im2", dataB);
    //readFile("out_1", dataD);

    parseToComplex(A, dataA);
    parseToComplex(B, dataB);

    double startTime = MPI_Wtime();

    c_fft2d_serial(A);
    c_fft2d_serial(B);

    pointMul_serial(A, B);

    c_inv_fft2d_serial(A);

    double endTime = MPI_Wtime();

    printf("Duration = %f\n", (endTime - startTime));

    float D[N][N];
    parseToReal(D, A);

    //print2DMatrix(D);

    checkIfEqual(D, dataD);
}
