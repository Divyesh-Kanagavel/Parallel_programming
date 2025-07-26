// passing argument to a thread

#include <unistd.h>
#include <pthread.h>
#include <stdio.h>

void* func1(void* arg) // void* because it is generic
{
    int *val = (int * )arg;
    for(int i=0;i<10;i++)
    {
        sleep(1);
        printf("func1 : %d %d \n", i, *val);
        (*val)++;
    }
    return NULL;

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
    int arg = 5;
    pthread_create(&thread1, NULL, func1, &arg);
    func2();
    pthread_join(thread1, NULL);
    printf("arg = %d\n", arg);
    return 0;
}