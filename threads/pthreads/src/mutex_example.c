#include <stdio.h>
#include <pthread.h>
#include <stdio.h>

int g_count = 0;
pthread_mutex_t my_mutex; // mutex 
void* add_count()
{
    pthread_mutex_lock(&my_mutex);
    g_count++; // critical section
    pthread_mutex_unlock(&my_mutex);
    return NULL;
}

int main()
{
    pthread_t thread1;
    pthread_t thread2;
    pthread_mutex_init(&my_mutex, NULL);
    pthread_create(&thread1, NULL, add_count, NULL);
    pthread_create(&thread2, NULL, add_count, NULL);
    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);
    printf("g_count = %d\n", g_count);
    return 0;
}