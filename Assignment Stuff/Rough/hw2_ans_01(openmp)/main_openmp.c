#include <stdio.h>
#include <omp.h>

int main_()
{
    int tid, k=0;
    tid = omp_get_thread_num();
        printf("Thread = %d\n", tid);
        k++;
    #pragma omp parallel num_threads(4) private(tid)
    {
        tid = omp_get_thread_num();
        printf("Thread = %d\n", tid);
        k++;
    }
    printf("OVERRR!, k = %d\n", k);
}

