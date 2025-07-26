#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

/*
unlike std::thread there is no joinable() API to check if thread is joined
to prevent undefined behaviour, we use struct to hold a flag - is_joined
*/

typedef struct thread_t
{
    pthread_t thread;
    int is_joined;
} thread_joinable;

void* print(void *arg)
{
    int *x = (int *)arg;
    while (((*x)--) > 0)
    {
        printf("x = %d\n", *x);
    }
    return NULL;
}

int main()
{
    printf("inside main()\n");
    thread_joinable thread1;

    thread1.is_joined = 0;
    int x = 10;
    pthread_create(&thread1.thread, NULL, print, (void*)&x);
    if (thread1.is_joined == 0)
    {
        //pthread_join(thread1.thread, NULL);
        pthread_detach(thread1.thread);
        thread1.is_joined = 1;
    }
    printf("inside main() after thread call\n");
}