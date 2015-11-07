#include <stdio.h>
#include <pthread.h>

#define T_COUNT 100

typedef int (*fpt)(int, int);

int lol(int *a)
{
    printf("Thread %d executing\n", *a);
    return *a-2*7/6+9-4;
}

int main()
{
    fpt lolptr = lol;
    pthread_t id[T_COUNT];
    int i, k = 0;

    for (i = 0; i < T_COUNT; i++)
    {
        k++;
        pthread_create(id + i, NULL, lolptr, &k);
    }

    for (i = 0; i < T_COUNT; i++)
    {
        pthread_join(id[i], NULL);
    }
}

