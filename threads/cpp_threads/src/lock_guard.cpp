// a lightweight wrapper on top of mutex
// scope based mutex lock and mutex unlocking.
// no explicit way to unlock lock guard
// no copy creatable of lock guard

#include <iostream>
#include <thread>
#include <mutex>

using namespace std;
int buffer = 0;
std::mutex m1;

void inc_buffer(const char* threadID, int loopFor)
{
    std::lock_guard<mutex> lock(m1); // mutex lock over the critical section. the unlocking after the std::lock_guard goes out of scope
    for(int i=0;i<loopFor;i++)
    {
        buffer++;
        cout << "thread number is : " << threadID << " " << buffer << std::endl;
    }
}

int main()
{
    thread t1(inc_buffer, "T0", 10);
    thread t2(inc_buffer, "T1", 10);
    t1.join();
    t2.join();
    return 0;
}