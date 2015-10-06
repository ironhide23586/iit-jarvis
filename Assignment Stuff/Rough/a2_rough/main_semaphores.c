#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>

#define T_COUNT 200

void *producer(int *);
void *consumer(int *);

int val = 0;

sem_t full, empty;

pthread_t prod_id, cons_id;

int main_semaphores()
{
    pthread_t id_prod[T_COUNT];
    pthread_t id_cons[T_COUNT];

    sem_init(&full, 1, 0);
    sem_init(&empty, 1, 10);

    int i;

    for (i = 0; i < T_COUNT; i++)
    {
        pthread_create(id_prod + i, NULL, producer, &val);
        pthread_create(id_cons + i, NULL, consumer, &val);
    }

    for (i = 0; i < T_COUNT; i++)
    {
        pthread_join(id_prod[i], NULL);
        pthread_join(id_cons[i], NULL);
    }
}

void *producer(int *arg)
{
    sem_wait(&empty);
    printf("Increasing val...\n");
    val++;
    printf("Val = %d\n", val);
    sem_post(&full);
}

void *consumer(int *arg)
{
    sem_wait(&full);
    printf("Decreasing val...\n");
    val--;
    printf("Val = %d\n", val);
    sem_post(&empty);
}
