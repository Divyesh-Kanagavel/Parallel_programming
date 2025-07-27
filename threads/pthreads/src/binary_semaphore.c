#include <stdio.h>
// semaphore is a signalling mechanism through which threads communicate and synchronize
// in this example,we see binary_semaphore

#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>
#include <dispatch/dispatch.h>

dispatch_semaphore_t smph_MainToThread;
dispatch_semaphore_t smph_ThreadToMain;

void* func(void* arg)
{
    dispatch_semaphore_wait(smph_MainToThread, DISPATCH_TIME_FOREVER); // tries to acquire the semaphore
    printf("inside func() \n");
    sleep(2); // sleep for 2 seconds
    dispatch_semaphore_signal(smph_ThreadToMain); // release the ThreadToMain semaphore for use by other threads
    return NULL;

}

int main()
{
    printf("inside main() \n");
    smph_MainToThread = dispatch_semaphore_create(0); // initially locked
    smph_ThreadToMain = dispatch_semaphore_create(0);

    pthread_t t;
    pthread_create(&t, NULL, func, NULL);
    printf("send signals from main to t\n");
    dispatch_semaphore_signal(smph_MainToThread); // this semaphore is released, so the func in thread t can now acqurie this semaphore
   printf("in main, trying to acquire ThreadToMain \n");
    dispatch_semaphore_wait(smph_ThreadToMain, DISPATCH_TIME_FOREVER);
   printf("got the signal \n");
    pthread_join(t, NULL);
    return 0;

}