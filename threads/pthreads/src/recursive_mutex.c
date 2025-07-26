// for functions with recursion, there is a separate class of mutex called the
// recursion mutex. which creates locks for each call of recursion and unlocks as stack unwinds

#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>

int buffer = 0;
pthread_mutex_t my_mutex; // global recursive mutex variable

void* recursion(void* arg)
{
    int* loopFor = (int *)arg;
    if (*loopFor < 0)
        return NULL;
    pthread_mutex_lock(&my_mutex);
    printf("buffer = %d\n" ,buffer++);
    (*loopFor)--;
    recursion((void *)loopFor);
    pthread_mutex_unlock(&my_mutex);
    printf("unlocked from thread id = %lu\n", (unsigned long)pthread_self);
    return NULL;

}

int main()
{
    pthread_mutexattr_t Attr;
    pthread_mutexattr_init(&Attr);
    pthread_mutexattr_settype(&Attr, PTHREAD_MUTEX_RECURSIVE); // recursive mutex type set
    pthread_mutex_init(&my_mutex, &Attr);

   int a = 2;
   pthread_t t1;
   pthread_t t2;
   pthread_create(&t1, NULL, recursion,(void *) &a);
   pthread_create(&t2, NULL, recursion, (void *) &a);
   pthread_join(t1, NULL);
   pthread_join(t2, NULL);

    return 0;
}