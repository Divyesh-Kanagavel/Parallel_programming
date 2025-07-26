#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
 #include <time.h>
typedef unsigned long long ull;

typedef struct args
{
    ull start;
    ull end;
    ull sum;
} args_t;

void* findEvenSum(void* arg)
{
    args_t* l_arg = (args_t *)arg;
    ull idx;
    l_arg->sum = 0;
    for(idx = l_arg->start; idx < l_arg->end;idx++)
    {
        if ((idx & 1) == 0)
            l_arg->sum += idx;
    }
    return NULL;
}

void* findOddSum(void* arg)
{
    args_t* l_arg = (args_t *)arg;
    ull idx;
    l_arg->sum = 0;
    for(idx = l_arg->start; idx < l_arg->end;idx++)
    {
        if ((idx & 1) == 1)
            l_arg->sum += idx;
    }
    return NULL;
}

int main()
{
    ull start = 0;
    ull end = 100000000;

    args_t args1;
    args_t args2;

    args1.start = start;
    args1.end = end;

    args2.start = start;
    args2.end = end;

    pthread_t thread1;
    pthread_t thread2;
    struct timespec start_t, end_t;
    clock_gettime(CLOCK_MONOTONIC, &start_t); // use this api to measure wall clock time instead of cpu time
    findEvenSum((void*)&args1);
    findOddSum((void*)&args2);
    clock_gettime(CLOCK_MONOTONIC, &end_t);

    printf("even sum is %llu\n", args1.sum);
    printf("odd sum is %llu\n", args2.sum);

    double time_taken = (end_t.tv_sec - start_t.tv_sec) + 
                    (end_t.tv_nsec - start_t.tv_nsec) / 1e9;
    printf("Execution time: %f seconds\n", time_taken);

    return 0;
}