// pthread example with argument returned from the function

// passing argument to a thread

#include <unistd.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

void* func1(void* arg) // void* because it is generic
{
    int *val = (int *)malloc(sizeof(int));
    *val = 0;
    for(int i=0;i<10;i++)
    {
        sleep(1);
        printf("func1 : %d %d \n", i, *val);
        (*val)++;
    }
    return val;

}

void func2()
{
    for(int i=0;i<5;i++)
    {
        sleep(1);
        printf("func2 :  %d\n", i);
    }
}

int main()
{
    pthread_t thread1;
    int* result; // argument returned from the function will be stored here.
    pthread_create(&thread1, NULL, func1, NULL);
    func2();
    pthread_join(thread1, (void *)&result);
    printf("arg = %d\n", *result);
    free(result);
    return 0;
}