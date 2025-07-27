// this is a classic problem solved by using threads
// there is a buffer of fixed size, which gets populated by a function - the producer
// then there is the consumer - which takes in data from the buffer

#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

const int MAX_SIZE = 100;
const int BUF_SIZE = 50;
int queue[MAX_SIZE]; // circular buffer
int front = 0; // front of the queue
int rear = 0;

pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cv = PTHREAD_COND_INITIALIZER;
void* producer(void* arg)
{
    int *value = (int *)arg;
    while(*value)
    {
        pthread_mutex_lock(&m);
        while (((rear+1)%BUF_SIZE) == front) // buffer is full
        {
            pthread_cond_wait(&cv, &m);
        }

        printf("producer : %d\n", *value);
        queue[rear] = (*value)--;
        rear = (rear + 1) % BUF_SIZE;
        pthread_cond_signal(&cv);
        pthread_mutex_unlock(&m);

    }
    return NULL;
}

void* consumer()
{
    while(1)
    {
        pthread_mutex_lock(&m);
        while (rear == front) // buffer is empty
        {
            pthread_cond_wait(&cv, &m);
        }

        int value = queue[front];
        front = (front + 1) % BUF_SIZE;
        printf("consumer : %d\n",value);
        pthread_cond_signal(&cv);
        pthread_mutex_unlock(&m);

        if (value == 1) break;
    }
    return NULL;
}

int main()
{
    pthread_t t1;
    pthread_t t2;
    int value = MAX_SIZE;
    pthread_create(&t1, NULL, producer, (void *)&value);
    pthread_create(&t2, NULL, consumer, NULL);
    pthread_join(t1, NULL);
    pthread_join(t2, NULL);
    return 0;
}





