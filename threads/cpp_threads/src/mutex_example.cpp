#include <iostream>
#include <thread>
#include <mutex>

using namespace std;
int g_count = 0; // global variable available to both threads
std::mutex m; // mutual exclusion lock

void add_count()
{
    m.lock();
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