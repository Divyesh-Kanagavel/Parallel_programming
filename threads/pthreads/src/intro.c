#include <pthread.h>
#include <unistd.h>
#include <stdio.h>

void* myturn(void* arg)
{
    for(int i=0;i<10;i++)
    {
        sleep(1);
        printf("my turn! %d \n", i);
    }
    return NULL;
}

void yourturn()
{
    for(int i=0;i<5;i++)
    {
        sleep(2);
        printf("your turn! %d \n", i);
    }
}


int main()
{
    pthread_t thread1;
    pthread_create(&thread1, NULL, myturn, NULL);
    yourturn();
    pthread_join(thread1,NULL);
    return 0;
}