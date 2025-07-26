// timed mutexes are similar to mutex's try_lock. but, a time component is added
// if a resource is locked by mutex, the other thread will wait for given time period for the resource. else will continue

#include <iostream>
#include <thread>
#include <mutex>
#include<chrono>
using namespace std;
int g_count = 0;
std::timed_mutex m;

void increment(int i)
{
    if (m.try_lock_for(chrono::seconds(2))) // you also have try_lock_until where you give time till when you want to wait. 
    {
        ++g_count;
        this_thread::sleep_for(chrono::seconds(1));
        std::cout << "Thread id = " << i << std::endl;
        m.unlock();
    }
    else{
       std::cout << "Thread id = " << i << std::endl;
    }
}

int main()
{
    thread t1(increment,1);
    thread t2(increment, 2);
    t1.join();
    t2.join();
    cout << "g_count = " << g_count << endl;
    return 0;
}