// there is another method to communicate between threads and synchronize between 
// them without a global variable. data from one thread or reference can be passed to spawned thread
// and it can be retrieved using future where the main thread till the spawned thread will pass the value to the main thread
#include <iostream>
#include <thread>
#include <future>
typedef unsigned long long ull;
using namespace std;

void findOddSum(std::promise<ull> objSum, ull start, ull end)
{
    ull sum = 0;
    for(ull idx = start ; idx < end; idx++)
    {
        if ((idx & 1) == 1)
            sum += idx;
           
    }
    objSum.set_value(sum); // value being set in the promise sent as argument
}

int main()
{
    std::promise<ull> p; // promise which is sent to the worker thread
    std::future<ull> f = p.get_future(); // future which retrieves the value set in the worker thread through promise

    thread t(findOddSum, std::move(p), 0, 100000);
    ull objSum = f.get(); // getting the value with thread sync , waits till the thread is done computing the sum

    std::cout << "result from objSum = " << objSum << std::endl;
    t.join();
    return 0;


}
