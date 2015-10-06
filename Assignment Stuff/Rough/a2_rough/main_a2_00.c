#include <stdio.h>
#include <pthread.h>

#define threads(n) *(threads + n)

void *func(int *arg)
{
    printf("Hello from thread %d\n", *arg);
}

int main()
{
    int n=9, k = 0, i;
    //scanf("No. of threads to run = %d", &n);
    pthread_t *threads;
    threads = (pthread_t *)malloc(n * sizeof(pthread_t));

    int *ar;

    ar = (int *)malloc(n * sizeof(int));

    for(i = 0; i < n; i++)
    {
        *(ar + i) = i;
    }

    for(i = 0; i < n; i++)
    {
        pthread_create(&threads(i), NULL, func, (ar + i));
        k++;
    }

    printf("%d\n", sizeof(threads));

    for(i = 0; i < n; i++)
    {
        pthread_join(threads(i), NULL);
    }

    free(threads);

    printf("%d\n", sizeof(threads));

    return 0;
}
