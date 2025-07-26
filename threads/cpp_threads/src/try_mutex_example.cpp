#include <iostream>
#include <thread>
#include <mutex>

using namespace std;
int g_count = 0; // global variable available to both threads
std::mutex m; // mutual exclusion lock

void add_count()
{
    //m.lock();
    /* there is a very important difference between lock and try_lock
    during lock, if the critical section is locked by a thread, other threads 
    wait for the resource to be unlocked and then lock it for themselves.
    with try_lock, the other threads do not wait, if the resource is locked, they 
    skip over it and complete the function. this causes data to be lost*/
    if (m.try_lock())
        for(int i=0;i < 100000;i++)
            g_count++; // without mutex lock , g_count is either 1 or two depending on the race condition for this critical section/region
    m.unlock();
}

int main()
{
    thread t1(add_count);
    thread t2(add_count);
    t1.join();
    t2.join();
    std::cout << "g_count = " << g_count << endl;
    return 0;
}