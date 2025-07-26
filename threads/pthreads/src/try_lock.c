#include <stdio.h>
#include <pthread.h>
#include <stdio.h>

int g_count = 0;
pthread_mutex_t my_mutex; // mutex 
void* add_count()
{
    //pthread_mutex_lock(&my_mutex);
    /* there is a very important difference between lock and try_lock
    during lock, if the critical section is locked by a thread, other threads 
    wait for the resource to be unlocked and then lock it for themselves.
    with try_lock, the other threads do not wait, if the resource is locked, they 
    skip over it and complete the function. this causes data to be lost*/
    int result = pthread_mutex_trylock(&my_mutex);
    if (result == 0) // 0 if mutex is available, else returns EBUSY
        for(int i=0;i<100000;i++)
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