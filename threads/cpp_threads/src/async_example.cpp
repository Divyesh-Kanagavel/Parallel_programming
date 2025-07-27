// there is another mechanism to run multiple threads with synchronization
// similar to promise and future seen before.
// here , one could return value from the function in the worker thread and
// get it in the main thread using future. 
// std::async launches the thread as and when required either in eager mode - std::launch::async or lazy evaluation, that is call the function only when future requests the value - std::launch::deferred

#include <iostream>
#include <thread>
#include <future>
typedef unsigned long long ull;
using namespace std;

ull findOddSum(ull start , ull end)
{
    ull oddSum = 0;
    std::cout << "thread ID = " << this_thread::get_id() << std::endl;
    for(ull idx=start; idx < end; idx++)
    {
        if ((idx & 1)==1)
            oddSum += idx;
    }
    return oddSum;
}

int main()
{
    ull start = 0;
    ull end = 100000000;
    cout << "main thread id = " << this_thread::get_id() << endl;
    //std::future<ull> oddSum = std::async(launch::async, findOddSum,start, end);// eager execution
    std::future<ull> oddSum = std::async(launch::deferred, findOddSum,start, end);// lazy execution
    cout << "waiting for result!\n";
    cout << "oddSum = " << oddSum.get() << endl;
    return 0;
}