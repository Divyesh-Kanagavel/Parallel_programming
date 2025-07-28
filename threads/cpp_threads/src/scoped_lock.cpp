// scoped lock -> uses std::lock mechanism to prevent deadlocks and the mutexes get unlocked once it goes out of scope
#include <iostream>
#include <thread>
#include <mutex>

using namespace std;
mutex g_mutex1;

void processData(int i)
{
    //g_mutex1.lock();
    std::scoped_lock lock(g_mutex1);
    cout << "processing thread id = " << i << endl;

}

int main()
{
    const int num_threads= 25;
    thread threads[num_threads];
    for(int i=0;i<num_threads;i++)
    {
        threads[i] = thread(processData, i);
    }
    for(int i=0;i<num_threads;i++)
    {
        threads[i].join();
    }
    return 0;

}
