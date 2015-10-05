#include <stdio.h>
#include <omp.h>

int main()
{
    #pragma omp parallel num_threads(30)
    {
        int i;
        printf("Hello world\n");
        #pragma omp for
        for (i = 1; i <=4; i++)
        {
            printf("Iteration %d\n", i);
        }
        printf("Goodbye World!!");
    }
}

